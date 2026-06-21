/* board.c — Zchezz board: 64-bit BBs + magic bitboards (v3.13)
 *
 * Changes from original:
 *   • bb[12] is now uint64_t (was split int32_t lo/hi pairs)
 *   • Magic bitboard tables for rook + bishop (O(1) slider attacks)
 *   • board_is_attacked rewritten using magic lookups
 *   • gen_bishop / gen_rook replaced with magic-based generators
 *   • All behaviour/hash/Zobrist identical to original
 */

#include "board.h"
#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════
 * GLOBALS
 * ═══════════════════════════════════════════════════════════════════ */
UndoFrame g_undo[STACK_SIZE];
int       g_undo_top = 0;

/* Default global NNUE accumulator (v3.13).
 * Used by main thread — SMP helpers allocate their own on the heap.
 * Must be 32-byte aligned for AVX2 SIMD in nnue_eval.
 *
 * CRITICAL: acc_dirty MUST start as 1 so that nnue_eval_bb returns 0
 * until nnue_rebuild is explicitly called.  Zero-initialization would
 * leave acc_dirty=0, causing the forward pass to run on a zero
 * accumulator (garbage evaluations — this was the v3.13 regression). */
NnueAccum g_nnue_accum __attribute__((aligned(32))) = { .acc_dirty = 1,
                                                         .ext_dirty = {1, 1} };

/* ═══════════════════════════════════════════════════════════════════
 * ZOBRIST  (64-bit — single hash)
 * ═══════════════════════════════════════════════════════════════════ */
uint64_t ZR_tab   [32*64];
uint64_t ZR_ep    [8];
uint64_t ZR_castle[16];
uint64_t ZR_side;

static uint64_t _zr_state;
static uint64_t zr_next(void) {
    uint64_t s = _zr_state;
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    _zr_state = s; return s;
}
static void zobrist_init(void) {
    _zr_state = 0xdeadbeefcafeULL;
    for (int i = 0; i < 32*64; i++) ZR_tab[i] = zr_next();
    for (int i = 0; i < 8;  i++) ZR_ep[i]     = zr_next();
    for (int i = 0; i < 16; i++) ZR_castle[i] = zr_next();
    ZR_side = zr_next();
}

uint64_t board_compute_hash(const uint8_t *b, int turn, uint8_t ca, int ep) {
    uint64_t h = 0;
    for (int sq=0;sq<64;sq++){uint8_t p=b[sq];if(!p)continue;h^=ZR_tab[p*64+sq];}
    if (turn==COL_W) h^=ZR_side;
    if (ep>=0)       h^=ZR_ep[ep&7];
    h^=ZR_castle[ca]; return h;
}

/* ═══════════════════════════════════════════════════════════════════
 * LEAPER ATTACK TABLES
 * ═══════════════════════════════════════════════════════════════════ */
uint64_t NATK[64], KATK[64];
int8_t   NATK_ARR[64][8], KATK_ARR[64][8];
int      NATK_N[64], KATK_N[64];
uint8_t  DIST[64*64];

static const int8_t KND_DR[8]={-2,-2,-1,-1,1,1,2,2};
static const int8_t KND_DC[8]={-1,1,-2,2,-2,2,-1,1};
static const int8_t KID_DR[8]={-1,-1,-1,0,0,1,1,1};
static const int8_t KID_DC[8]={-1,0,1,-1,1,-1,0,1};

static void leaper_init(void) {
    for (int sq=0;sq<64;sq++) {
        int r=sq>>3,c=sq&7,nn=0,kn=0;
        for (int d=0;d<8;d++) {
            int nr=r+KND_DR[d],nc=c+KND_DC[d];
            if(nr>=0&&nr<8&&nc>=0&&nc<8){int s=nr*8+nc;NATK_ARR[sq][nn++]=(int8_t)s;NATK[sq]|=(uint64_t)1<<s;}
            nr=r+KID_DR[d];nc=c+KID_DC[d];
            if(nr>=0&&nr<8&&nc>=0&&nc<8){int s=nr*8+nc;KATK_ARR[sq][kn++]=(int8_t)s;KATK[sq]|=(uint64_t)1<<s;}
        }
        for(int i=nn;i<8;i++)NATK_ARR[sq][i]=-1;
        for(int i=kn;i<8;i++)KATK_ARR[sq][i]=-1;
        NATK_N[sq]=nn; KATK_N[sq]=kn;
    }
    for(int a=0;a<64;a++)for(int b2=0;b2<64;b2++){
        int dr=(a>>3)-(b2>>3),dc=(a&7)-(b2&7);
        if(dr<0)dr=-dr; if(dc<0)dc=-dc;
        DIST[(a<<6)|b2]=(uint8_t)(dr>dc?dr:dc);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MAGIC BITBOARDS
 *
 * Pre-computed magic numbers from public domain (Tord Romstad / others).
 * We use plain magic (not PEXT) so this works on any x86/ARM.
 * The attack tables are initialised once by magic_init().
 * ═══════════════════════════════════════════════════════════════════ */

/* Shared attack storage — rooks need up to 4096 entries per square,
 * bishops up to 512.  We allocate one flat array and point into it. */
static uint64_t _rook_atk_store[64*4096];
static uint64_t _bish_atk_store[64*512];

uint64_t *ROOK_TBL [64];
uint64_t  ROOK_MAGIC[64];
uint64_t  ROOK_MASK [64];
int       ROOK_SHIFT[64];

uint64_t *BISH_TBL [64];
uint64_t  BISH_MAGIC[64];
uint64_t  BISH_MASK [64];
int       BISH_SHIFT[64];

/* Pre-computed magic numbers (well-known, public domain) */
static const uint64_t ROOK_MAGICS[64] = {
    0xa8002c000108020ULL,0x6c00049b0002001ULL,0x100200010090040ULL,0x2480041000800801ULL,
    0x280028004000800ULL,0x900410008040022ULL,0x280020001001080ULL,0x2880002041000080ULL,
    0xa000800080400034ULL,0x4808020004000ULL,0x2290802004801000ULL,0x411000d00100020ULL,
    0x402800800040080ULL,0xb000401004208ULL,0x2409000100040200ULL,0x1002100004082ULL,
    0x22878001e24000ULL,0x1090810021004010ULL,0x801030040200012ULL,0x500808008001000ULL,
    0xa08018014000880ULL,0x8000808004000200ULL,0x201008080010200ULL,0x801020000441091ULL,
    0x800080204005ULL,0x1040200040100048ULL,0x120200402082ULL,0xd14880480100080ULL,
    0x12040280080080ULL,0x100040080020080ULL,0x9020010080800200ULL,0x813241200148449ULL,
    0x491604001800080ULL,0x100401000402001ULL,0x4820010021001040ULL,0x400402202000812ULL,
    0x209009005000802ULL,0x810800601800400ULL,0x4301083214000150ULL,0x204026458e001401ULL,
    0x40204000808000ULL,0x8001008040010020ULL,0x8410820820420010ULL,0x1003001000090020ULL,
    0x804040008008080ULL,0x12000810020004ULL,0x1000100200040208ULL,0x430000a044020001ULL,
    0x280009023410300ULL,0xe0100040002240ULL,0x200100401700ULL,0x2244100408008080ULL,
    0x8000400801980ULL,0x2000810040200ULL,0x8010100228810400ULL,0x2000009044210200ULL,
    0x4080008040102101ULL,0x40002080411d01ULL,0x2005524060000901ULL,0x502001008400422ULL,
    0x489a000810200402ULL,0x1004400080a13ULL,0x4000011008020084ULL,0x26002114058042ULL,
};
static const uint64_t BISH_MAGICS[64] = {
    0x89a1121896040240ULL,0x2004844802002010ULL,0x2068080051921000ULL,0x62880a0220200808ULL,
    0x4042004000000ULL,0x100822020200011ULL,0xc00444222012000aULL,0x28808801216001ULL,
    0x400492088408100ULL,0x201c401040c0084ULL,0x840800910a0010ULL,0x82080240060ULL,
    0x2000840504006000ULL,0x30010c4108405004ULL,0x1008005410080802ULL,0x8144042209100900ULL,
    0x208081020014400ULL,0x4800201208ca00ULL,0xf18140408012008ULL,0x1004002802102001ULL,
    0x841000820080811ULL,0x40200200a42008ULL,0x800054042000ULL,0x88010400410c9000ULL,
    0x520040470104290ULL,0x1004040051500081ULL,0x2002081833080021ULL,0x400c00c010142ULL,
    0x941408200c002000ULL,0x658810000806011ULL,0x188071040440a00ULL,0x4800404002011c00ULL,
    0x104442040404200ULL,0x511080202091021ULL,0x4022401120400ULL,0x80c0040400080120ULL,
    0x8040010040820802ULL,0x480810700020090ULL,0x102008e00040242ULL,0x809005202050100ULL,
    0x8002024220104080ULL,0x431008804142000ULL,0x19001802081400ULL,0x200014208040080ULL,
    0x3308082008200100ULL,0x41010500040c020ULL,0x4012020c04210308ULL,0x208220a202004080ULL,
    0x111040120082000ULL,0x6803040141280a00ULL,0x2101004202410000ULL,0x8200000041108022ULL,
    0x21082088000ULL,0x2410204010040ULL,0x40100400809000ULL,0x822088220820214ULL,
    0x40808090012004ULL,0x910224040218c9ULL,0x402814422015008ULL,0x90014004842410ULL,
    0x1000042304105ULL,0x10008830412a00ULL,0x2520081090008908ULL,0x40102000a0a60140ULL,
};

/* Build the occupancy mask for a rook on sq (all squares it can reach, edges excluded) */
static uint64_t rook_mask(int sq) {
    uint64_t m = 0;
    int r=sq>>3,c=sq&7,i;
    for(i=r-1;i>0;i--) m|=(uint64_t)1<<(i*8+c);
    for(i=r+1;i<7;i++) m|=(uint64_t)1<<(i*8+c);
    for(i=c-1;i>0;i--) m|=(uint64_t)1<<(r*8+i);
    for(i=c+1;i<7;i++) m|=(uint64_t)1<<(r*8+i);
    return m;
}
static uint64_t bish_mask(int sq) {
    uint64_t m = 0;
    int r=sq>>3,c=sq&7,nr,nc;
    nr=r-1;nc=c-1;while(nr>0&&nc>0){m|=(uint64_t)1<<(nr*8+nc);nr--;nc--;}
    nr=r-1;nc=c+1;while(nr>0&&nc<7){m|=(uint64_t)1<<(nr*8+nc);nr--;nc++;}
    nr=r+1;nc=c-1;while(nr<7&&nc>0){m|=(uint64_t)1<<(nr*8+nc);nr++;nc--;}
    nr=r+1;nc=c+1;while(nr<7&&nc<7){m|=(uint64_t)1<<(nr*8+nc);nr++;nc++;}
    return m;
}

/* Compute actual rook attacks for a given occupancy (slow, for table init only) */
static uint64_t rook_atk_slow(int sq, uint64_t occ) {
    uint64_t m=0; int r=sq>>3,c=sq&7,i;
    for(i=r-1;i>=0;i--){uint64_t b=(uint64_t)1<<(i*8+c);m|=b;if(occ&b)break;}
    for(i=r+1;i<=7;i++){uint64_t b=(uint64_t)1<<(i*8+c);m|=b;if(occ&b)break;}
    for(i=c-1;i>=0;i--){uint64_t b=(uint64_t)1<<(r*8+i);m|=b;if(occ&b)break;}
    for(i=c+1;i<=7;i++){uint64_t b=(uint64_t)1<<(r*8+i);m|=b;if(occ&b)break;}
    return m;
}
static uint64_t bish_atk_slow(int sq, uint64_t occ) {
    uint64_t m=0; int r=sq>>3,c=sq&7,nr,nc;
    nr=r-1;nc=c-1;while(nr>=0&&nc>=0){uint64_t b=(uint64_t)1<<(nr*8+nc);m|=b;if(occ&b)break;nr--;nc--;}
    nr=r-1;nc=c+1;while(nr>=0&&nc<=7){uint64_t b=(uint64_t)1<<(nr*8+nc);m|=b;if(occ&b)break;nr--;nc++;}
    nr=r+1;nc=c-1;while(nr<=7&&nc>=0){uint64_t b=(uint64_t)1<<(nr*8+nc);m|=b;if(occ&b)break;nr++;nc--;}
    nr=r+1;nc=c+1;while(nr<=7&&nc<=7){uint64_t b=(uint64_t)1<<(nr*8+nc);m|=b;if(occ&b)break;nr++;nc++;}
    return m;
}

/* Enumerate all subsets of mask (Carry-Rippler) and fill attack table */
static void fill_magic_table(int sq, uint64_t mask, uint64_t magic, int shift,
                              uint64_t *tbl,
                              uint64_t (*atk_fn)(int,uint64_t)) {
    uint64_t occ = 0;
    do {
        int idx = (int)((occ * magic) >> shift);
        tbl[idx] = atk_fn(sq, occ);
        occ = (occ - mask) & mask;   /* Carry-Rippler */
    } while (occ);
}

static void magic_init(void) {
    for (int sq=0;sq<64;sq++) {
        /* Rook */
        ROOK_MASK[sq]  = rook_mask(sq);
        ROOK_MAGIC[sq] = ROOK_MAGICS[sq];
        int rbits = __builtin_popcountll(ROOK_MASK[sq]);
        ROOK_SHIFT[sq] = 64 - rbits;
        ROOK_TBL[sq]   = _rook_atk_store + sq*4096;
        fill_magic_table(sq, ROOK_MASK[sq], ROOK_MAGIC[sq], ROOK_SHIFT[sq],
                         ROOK_TBL[sq], rook_atk_slow);

        /* Bishop */
        BISH_MASK[sq]  = bish_mask(sq);
        BISH_MAGIC[sq] = BISH_MAGICS[sq];
        int bbits = __builtin_popcountll(BISH_MASK[sq]);
        BISH_SHIFT[sq] = 64 - bbits;
        BISH_TBL[sq]   = _bish_atk_store + sq*512;
        fill_magic_table(sq, BISH_MASK[sq], BISH_MAGIC[sq], BISH_SHIFT[sq],
                         BISH_TBL[sq], bish_atk_slow);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MATERIAL + PIECE→BB INDEX
 * ═══════════════════════════════════════════════════════════════════ */
int MV_TAB[7] = { 0, 100, 322, 335, 500, 900, 20000 };

/* piece value → bb index (0..11) */
static const int8_t P2BI[32] = {
  -1,-1,-1,-1,-1,-1,-1,-1,
  -1, 0, 1, 2, 3, 4, 5,-1,   /* WP..WK */
  -1, 6, 7, 8, 9,10,11,-1,   /* BP..BK */
  -1,-1,-1,-1,-1,-1,-1,-1,
};

/* ── BB helpers ──────────────────────────────────────────────────── */
static inline int lsb64(uint64_t n) { return __builtin_ctzll(n); }

static inline void bb_set(Board *b, int bi, int sq) {
    b->bb[bi] |= (uint64_t)1 << sq;
}
static inline void bb_clr(Board *b, int bi, int sq) {
    b->bb[bi] &= ~((uint64_t)1 << sq);
}

static void rebuild_bb(Board *b) {
    memset(b->bb, 0, sizeof(b->bb));
    for (int sq=0;sq<64;sq++){
        uint8_t p=b->b[sq]; if(!p) continue;
        int bi=P2BI[p&31]; if(bi<0) continue;
        bb_set(b,bi,sq);
    }
    /* Rebuild cached occupancy */
    b->occ_w = b->bb[0]|b->bb[1]|b->bb[2]|b->bb[3]|b->bb[4]|b->bb[5];
    b->occ_b = b->bb[6]|b->bb[7]|b->bb[8]|b->bb[9]|b->bb[10]|b->bb[11];
    b->occ   = b->occ_w | b->occ_b;
}

/* ═══════════════════════════════════════════════════════════════════
 * board_init
 * ═══════════════════════════════════════════════════════════════════ */
void board_init(void) {
    zobrist_init();
    leaper_init();
    magic_init();
}

/* ═══════════════════════════════════════════════════════════════════
 * board_load_fen
 * ═══════════════════════════════════════════════════════════════════ */
int board_load_fen(Board *b, const char *fen) {
    memset(b,0,sizeof(*b));
    b->ep=-1; b->fm=1; b->wk=60; b->bk=4;
    static const char *SYMS="PNBRQKpnbrqk";
    static const uint8_t VALS[12]={WP,WN,WB,WR,WQ,WK,BP,BN,BB,BR,BQ,BK};
    int sq=0; const char *p=fen;
    while(*p&&*p!=' '){
        if(*p>='1'&&*p<='8'){sq+=(*p-'0');}
        else if(*p!='/'){const char*s=strchr(SYMS,*p);if(s&&sq<64)b->b[sq++]=VALS[s-SYMS];}
        p++;
    }
    if(*p==' ')p++;
    b->turn=(*p=='b')?COL_B:COL_W;
    while(*p&&*p!=' ')p++; if(*p==' ')p++;
    b->ca=0;
    while(*p&&*p!=' '){
        if(*p=='K')b->ca|=CA_WK; if(*p=='Q')b->ca|=CA_WQ;
        if(*p=='k')b->ca|=CA_BK; if(*p=='q')b->ca|=CA_BQ;
        p++;
    }
    if(*p==' ')p++;
    b->ep=-1;
    static const char *FILES="abcdefgh",*RANKS="87654321";
    if(*p&&*p!='-'){
        const char*ef=strchr(FILES,*p);
        if(ef&&*(p+1)){const char*er=strchr(RANKS,*(p+1));if(er)b->ep=(int)(er-RANKS)*8+(int)(ef-FILES);}
    }
    while(*p&&*p!=' ')p++; if(*p==' ')p++;
    if(*p&&*p!=' '){b->hm=(uint8_t)atoi(p);while(*p&&*p!=' ')p++;if(*p==' ')p++;}
    if(*p)b->fm=(uint16_t)atoi(p);
    for(int s=0;s<64;s++){if(b->b[s]==WK)b->wk=(uint8_t)s;if(b->b[s]==BK)b->bk=(uint8_t)s;}
    rebuild_bb(b);
    b->hash=board_compute_hash(b->b,b->turn,b->ca,b->ep);
    b->hist_len=0;
    board_bind_undo_global(b);  /* default: use global undo stack */
    board_bind_nnue_global(b);  /* default: use global NNUE accumulator (v3.13) */
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * board_to_fen
 * ═══════════════════════════════════════════════════════════════════ */
void board_to_fen(const Board *b, char *buf) {
    static const char PCMAP[23]={0,0,0,0,0,0,0,0,0,'P','N','B','R','Q','K',0,0,'p','n','b','r','q','k'};
    char *out=buf;
    for(int r=0;r<8;r++){
        int empty=0;
        for(int c=0;c<8;c++){uint8_t p=b->b[r*8+c];if(!p){empty++;}else{if(empty){*out++='0'+empty;empty=0;}*out++=PCMAP[p<23?p:0];}}
        if(empty)*out++='0'+empty;
        if(r<7)*out++='/';
    }
    *out++=' '; *out++=b->turn==COL_W?'w':'b'; *out++=' ';
    if(!b->ca)*out++='-';
    else{if(b->ca&CA_WK)*out++='K';if(b->ca&CA_WQ)*out++='Q';if(b->ca&CA_BK)*out++='k';if(b->ca&CA_BQ)*out++='q';}
    *out++=' ';
    if(b->ep<0){*out++='-';}else{*out++="abcdefgh"[b->ep&7];*out++="87654321"[b->ep>>3];}
    sprintf(out," %d %d",b->hm,b->fm);
}

/* ═══════════════════════════════════════════════════════════════════
 * board_is_attacked  — fast magic version
 * ═══════════════════════════════════════════════════════════════════ */
int board_is_attacked(const Board *b, int sq, int by) {
    /* Use cached occupancy — avoids 12-OR recomputation */
    uint64_t occ = b->occ;
    uint64_t sq_bb = (uint64_t)1 << sq;

    if (by == COL_W) {
        /* White pawn attacks: bitboard shift (Phase 2) */
        if (bpawn_attacks_bb(sq_bb) & b->bb[0]) return 1;  /* sq attacked by WP */
        if (NATK[sq] & b->bb[1])  return 1;   /* white knight */
        if (bish_attacks(sq,occ) & (b->bb[2]|b->bb[4])) return 1;  /* WB|WQ */
        if (rook_attacks(sq,occ) & (b->bb[3]|b->bb[4])) return 1;  /* WR|WQ */
        if (KATK[sq] & b->bb[5])  return 1;   /* white king */
    } else {
        /* Black pawn attacks: bitboard shift (Phase 2) */
        if (wpawn_attacks_bb(sq_bb) & b->bb[6]) return 1;  /* sq attacked by BP */
        if (NATK[sq] & b->bb[7])  return 1;
        if (bish_attacks(sq,occ) & (b->bb[8]|b->bb[10])) return 1; /* BB|BQ */
        if (rook_attacks(sq,occ) & (b->bb[9]|b->bb[10])) return 1; /* BR|BQ */
        if (KATK[sq] & b->bb[11]) return 1;
    }
    return 0;
}

int board_in_check(const Board *b) {
    int ksq = b->turn==COL_W ? b->wk : b->bk;
    return board_is_attacked(b, ksq, b->turn^24);
}

/* ── board_checkers: bitboard of all pieces giving check to STM king ── */
uint64_t board_checkers(const Board *b) {
    int ksq = b->turn==COL_W ? b->wk : b->bk;
    int opp = b->turn ^ 24;
    uint64_t occ = b->occ;
    uint64_t checkers = 0;
    if (opp == COL_W) {
        checkers |= bpawn_attacks_bb((uint64_t)1 << ksq) & b->bb[0];
        checkers |= NATK[ksq] & b->bb[1];
        checkers |= bish_attacks(ksq, occ) & (b->bb[2] | b->bb[4]);
        checkers |= rook_attacks(ksq, occ) & (b->bb[3] | b->bb[4]);
    } else {
        checkers |= wpawn_attacks_bb((uint64_t)1 << ksq) & b->bb[6];
        checkers |= NATK[ksq] & b->bb[7];
        checkers |= bish_attacks(ksq, occ) & (b->bb[8] | b->bb[10]);
        checkers |= rook_attacks(ksq, occ) & (b->bb[9] | b->bb[10]);
    }
    return checkers;
}

/* ── board_pinned: bitboard of STM pieces pinned to STM king ── */
uint64_t board_pinned(const Board *b) {
    int ksq = b->turn==COL_W ? b->wk : b->bk;
    uint64_t occ = b->occ;
    uint64_t my  = b->turn==COL_W ? b->occ_w : b->occ_b;
    uint64_t pinned = 0;

    /* Find opponent sliders that could pin through our pieces */
    uint64_t opp_bq, opp_rq;
    if (b->turn == COL_W) {
        opp_bq = b->bb[8] | b->bb[10];  /* BB + BQ */
        opp_rq = b->bb[9] | b->bb[10];  /* BR + BQ */
    } else {
        opp_bq = b->bb[2] | b->bb[4];   /* WB + WQ */
        opp_rq = b->bb[3] | b->bb[4];   /* WR + WQ */
    }

    /* Bishop/queen X-ray: fire bishop attack from king with NO blockers,
     * intersect with opponent bishop/queen to find potential pinners */
    uint64_t diag_pinners = bish_attacks(ksq, 0) & opp_bq;
    while (diag_pinners) {
        int pinner_sq = __builtin_ctzll(diag_pinners);
        diag_pinners &= diag_pinners - 1;
        /* Pieces between king and pinner */
        uint64_t between = bish_attacks(ksq, occ) & bish_attacks(pinner_sq, occ);
        /* The "between" mask from magic attacks doesn't include the endpoints,
         * so we need to look at what's between them on the actual ray.
         * Simpler: use the intersection of king's X-ray and pinner's X-ray. */
        /* Actually, let's use the standard approach: get the ray between
         * king and pinner, count our pieces on it. */
        uint64_t occ_no_pinner = occ & ~((uint64_t)1 << pinner_sq);
        uint64_t ray = bish_attacks(ksq, occ_no_pinner) & bish_attacks(pinner_sq, occ_no_pinner);
        uint64_t blockers = ray & my;
        if (blockers && !(blockers & (blockers - 1))) {
            /* Exactly one of our pieces blocks — it's pinned */
            pinned |= blockers;
        }
    }

    /* Rook/queen X-ray */
    uint64_t orth_pinners = rook_attacks(ksq, 0) & opp_rq;
    while (orth_pinners) {
        int pinner_sq = __builtin_ctzll(orth_pinners);
        orth_pinners &= orth_pinners - 1;
        uint64_t occ_no_pinner = occ & ~((uint64_t)1 << pinner_sq);
        uint64_t ray = rook_attacks(ksq, occ_no_pinner) & rook_attacks(pinner_sq, occ_no_pinner);
        uint64_t blockers = ray & my;
        if (blockers && !(blockers & (blockers - 1))) {
            pinned |= blockers;
        }
    }

    return pinned;
}

/* ── board_pin_ray: given a pinned square, return the full ray (king..pinner) ── */
uint64_t board_pin_ray(const Board *b, int pinned_sq) {
    int ksq = b->turn==COL_W ? b->wk : b->bk;
    uint64_t occ = b->occ;
    uint64_t pin_bb = (uint64_t)1 << pinned_sq;

    /* Determine direction: diagonal or orthogonal */
    uint64_t opp_bq, opp_rq;
    if (b->turn == COL_W) {
        opp_bq = b->bb[8] | b->bb[10];
        opp_rq = b->bb[9] | b->bb[10];
    } else {
        opp_bq = b->bb[2] | b->bb[4];
        opp_rq = b->bb[3] | b->bb[4];
    }

    /* Check if pinned on diagonal */
    uint64_t diag_from_king = bish_attacks(ksq, pin_bb);  /* ray stops at pinned piece */
    /* Fire from pinned piece outward (away from king) */
    uint64_t diag_from_pin  = bish_attacks(pinned_sq, occ);
    /* Pinner = opponent B/Q on the ray from king through pin */
    uint64_t diag_pinner = diag_from_pin & opp_bq & bish_attacks(ksq, 0);
    if (diag_pinner) {
        int pinner_sq = __builtin_ctzll(diag_pinner);
        /* Ray = squares between king and pinner, plus the pinner itself */
        uint64_t occ_just_endpoints = ((uint64_t)1 << ksq) | ((uint64_t)1 << pinner_sq);
        return (bish_attacks(ksq, occ_just_endpoints) & bish_attacks(pinner_sq, occ_just_endpoints))
               | ((uint64_t)1 << pinner_sq);
    }

    /* Check if pinned on orthogonal */
    uint64_t orth_from_pin = rook_attacks(pinned_sq, occ);
    uint64_t orth_pinner = orth_from_pin & opp_rq & rook_attacks(ksq, 0);
    if (orth_pinner) {
        int pinner_sq = __builtin_ctzll(orth_pinner);
        uint64_t occ_just_endpoints = ((uint64_t)1 << ksq) | ((uint64_t)1 << pinner_sq);
        return (rook_attacks(ksq, occ_just_endpoints) & rook_attacks(pinner_sq, occ_just_endpoints))
               | ((uint64_t)1 << pinner_sq);
    }

    return ~(uint64_t)0;  /* not pinned — unrestricted */
}

/* ═══════════════════════════════════════════════════════════════════
 * MOVE GENERATION
 * ═══════════════════════════════════════════════════════════════════ */
#define PUSH(f,t,pr,ep,cs) do { \
    Move *_m=&out[n++]; \
    _m->from=(f);_m->to=(t);_m->prom=(pr);_m->epc=(ep);_m->castle=(cs);_m->score=0; \
} while(0)

#define PAWN_PROMOS(f,t) do { \
    PUSH(f,t,5,0,0); PUSH(f,t,4,0,0); PUSH(f,t,3,0,0); PUSH(f,t,2,0,0); \
} while(0)

int board_gen_moves(const Board *b, Move *out) {
    int n=0;
    const uint8_t *bd=b->b;
    int col=b->turn, op=col^24;
    int ep=b->ep;

    /* Use cached occupancy (Phase 1) */
    uint64_t occ = b->occ;
    uint64_t my   = col==COL_W ? b->occ_w : b->occ_b;
    uint64_t them = col==COL_W ? b->occ_b : b->occ_w;

    if(col==COL_W){
        /* White pawns */
        { uint64_t wp=b->bb[0]; uint64_t tmp=wp;
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            int r=sq>>3,f=sq&7;
            int s1=sq-8;
            if(s1>=0&&!bd[s1]){
                if(r==1){PAWN_PROMOS(sq,s1);}
                else{PUSH(sq,s1,0,0,0);if(r==6&&!bd[sq-16])PUSH(sq,sq-16,0,0,0);}
            }
            if(f>0){int c2=sq-9;if(c2>=0&&bd[c2]&&PC_COLOR(bd[c2])==COL_B){if(r==1)PAWN_PROMOS(sq,c2);else PUSH(sq,c2,0,0,0);}}
            if(f<7){int c2=sq-7;if(c2>=0&&bd[c2]&&PC_COLOR(bd[c2])==COL_B){if(r==1)PAWN_PROMOS(sq,c2);else PUSH(sq,c2,0,0,0);}}
            if(ep>=0&&r==3){if(f>0&&ep==sq-9)PUSH(sq,ep,0,1,0);if(f<7&&ep==sq-7)PUSH(sq,ep,0,1,0);}
          }
        }
        /* White knights */
        { uint64_t tmp=b->bb[1];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=NATK[sq]&~my;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* White bishops */
        { uint64_t tmp=b->bb[2];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=bish_attacks(sq,occ)&~my;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* White rooks */
        { uint64_t tmp=b->bb[3];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=rook_attacks(sq,occ)&~my;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* White queens */
        { uint64_t tmp=b->bb[4];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=queen_attacks(sq,occ)&~my;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* White king */
        { int sq=b->wk;
          uint64_t atk=KATK[sq]&~my;
          while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          if(sq==60){
            if((b->ca&CA_WK)&&bd[63]==WR&&!bd[61]&&!bd[62]&&
               !board_is_attacked(b,60,COL_B)&&!board_is_attacked(b,61,COL_B)&&!board_is_attacked(b,62,COL_B))
              PUSH(60,62,0,0,1);
            if((b->ca&CA_WQ)&&bd[56]==WR&&!bd[59]&&!bd[58]&&!bd[57]&&
               !board_is_attacked(b,60,COL_B)&&!board_is_attacked(b,59,COL_B)&&!board_is_attacked(b,58,COL_B))
              PUSH(60,58,0,0,2);
          }
        }
    } else {
        /* Black pawns */
        { uint64_t tmp=b->bb[6];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            int r=sq>>3,f=sq&7;
            int s1=sq+8;
            if(s1<64&&!bd[s1]){
                if(r==6){PAWN_PROMOS(sq,s1);}
                else{PUSH(sq,s1,0,0,0);if(r==1&&!bd[sq+16])PUSH(sq,sq+16,0,0,0);}
            }
            if(f>0){int c2=sq+7;if(c2<64&&bd[c2]&&PC_COLOR(bd[c2])==COL_W){if(r==6)PAWN_PROMOS(sq,c2);else PUSH(sq,c2,0,0,0);}}
            if(f<7){int c2=sq+9;if(c2<64&&bd[c2]&&PC_COLOR(bd[c2])==COL_W){if(r==6)PAWN_PROMOS(sq,c2);else PUSH(sq,c2,0,0,0);}}
            if(ep>=0&&r==4){if(f>0&&ep==sq+7)PUSH(sq,ep,0,1,0);if(f<7&&ep==sq+9)PUSH(sq,ep,0,1,0);}
          }
        }
        /* Black knights */
        { uint64_t tmp=b->bb[7];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=NATK[sq]&~my;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Black bishops */
        { uint64_t tmp=b->bb[8];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=bish_attacks(sq,occ)&~my;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Black rooks */
        { uint64_t tmp=b->bb[9];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=rook_attacks(sq,occ)&~my;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Black queens */
        { uint64_t tmp=b->bb[10];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=queen_attacks(sq,occ)&~my;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Black king */
        { int sq=b->bk;
          uint64_t atk=KATK[sq]&~my;
          while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          if(sq==4){
            if((b->ca&CA_BK)&&bd[7]==BR&&!bd[5]&&!bd[6]&&
               !board_is_attacked(b,4,COL_W)&&!board_is_attacked(b,5,COL_W)&&!board_is_attacked(b,6,COL_W))
              PUSH(4,6,0,0,3);
            if((b->ca&CA_BQ)&&bd[0]==BR&&!bd[3]&&!bd[2]&&!bd[1]&&
               !board_is_attacked(b,4,COL_W)&&!board_is_attacked(b,3,COL_W)&&!board_is_attacked(b,2,COL_W))
              PUSH(4,2,0,0,4);
          }
        }
    }
    return n;
}

/* Captures + promotions only (qsearch) */
int board_gen_captures(const Board *b, Move *out) {
    int n=0;
    const uint8_t *bd=b->b;
    int col=b->turn, op=col^24;
    int ep=b->ep;

    /* Use cached occupancy (Phase 1) */
    uint64_t occ = b->occ;
    uint64_t my   = col==COL_W ? b->occ_w : b->occ_b;
    uint64_t them = col==COL_W ? b->occ_b : b->occ_w;

    if(col==COL_W){
        /* Pawn caps + promos */
        { uint64_t tmp=b->bb[0];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            int r=sq>>3,f=sq&7;
            if(r==1){int s1=sq-8;if(s1>=0&&!bd[s1]){PAWN_PROMOS(sq,s1);}}
            if(f>0){int c2=sq-9;if(c2>=0&&bd[c2]&&PC_COLOR(bd[c2])==COL_B){if(r==1)PAWN_PROMOS(sq,c2);else PUSH(sq,c2,0,0,0);}}
            if(f<7){int c2=sq-7;if(c2>=0&&bd[c2]&&PC_COLOR(bd[c2])==COL_B){if(r==1)PAWN_PROMOS(sq,c2);else PUSH(sq,c2,0,0,0);}}
            if(ep>=0&&r==3){if(f>0&&ep==sq-9)PUSH(sq,ep,0,1,0);if(f<7&&ep==sq-7)PUSH(sq,ep,0,1,0);}
          }
        }
        /* Knight caps */
        { uint64_t tmp=b->bb[1];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=NATK[sq]&them;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Bishop caps */
        { uint64_t tmp=b->bb[2];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=bish_attacks(sq,occ)&them;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Rook caps */
        { uint64_t tmp=b->bb[3];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=rook_attacks(sq,occ)&them;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Queen caps */
        { uint64_t tmp=b->bb[4];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=queen_attacks(sq,occ)&them;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* King caps */
        { int sq=b->wk; uint64_t atk=KATK[sq]&them;
          while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
        }
    } else {
        { uint64_t tmp=b->bb[6];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            int r=sq>>3,f=sq&7;
            if(r==6){int s1=sq+8;if(s1<64&&!bd[s1]){PAWN_PROMOS(sq,s1);}}
            if(f>0){int c2=sq+7;if(c2<64&&bd[c2]&&PC_COLOR(bd[c2])==COL_W){if(r==6)PAWN_PROMOS(sq,c2);else PUSH(sq,c2,0,0,0);}}
            if(f<7){int c2=sq+9;if(c2<64&&bd[c2]&&PC_COLOR(bd[c2])==COL_W){if(r==6)PAWN_PROMOS(sq,c2);else PUSH(sq,c2,0,0,0);}}
            if(ep>=0&&r==4){if(f>0&&ep==sq+7)PUSH(sq,ep,0,1,0);if(f<7&&ep==sq+9)PUSH(sq,ep,0,1,0);}
          }
        }
        { uint64_t tmp=b->bb[7];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=NATK[sq]&them;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        { uint64_t tmp=b->bb[8];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=bish_attacks(sq,occ)&them;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        { uint64_t tmp=b->bb[9];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=rook_attacks(sq,occ)&them;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        { uint64_t tmp=b->bb[10];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=queen_attacks(sq,occ)&them;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        { int sq=b->bk; uint64_t atk=KATK[sq]&them;
          while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
        }
    }
    return n;
}

/* Quiet moves only (non-captures, non-promotions, includes castles) — for staged movegen */
int board_gen_quiets(const Board *b, Move *out) {
    int n=0;
    const uint8_t *bd=b->b;
    int col=b->turn;
    uint64_t occ = b->occ;
    uint64_t my  = col==COL_W ? b->occ_w : b->occ_b;
    uint64_t them= col==COL_W ? b->occ_b : b->occ_w;
    uint64_t empty = ~occ;

    if(col==COL_W){
        /* Pawn pushes (non-promotion) */
        { uint64_t tmp=b->bb[0];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            int r=sq>>3,f=sq&7;
            int s1=sq-8;
            if(s1>=0&&!bd[s1]&&r!=1){
                PUSH(sq,s1,0,0,0);
                if(r==6&&!bd[sq-16]) PUSH(sq,sq-16,0,0,0);
            }
          }
        }
        /* Knights — non-capture */
        { uint64_t tmp=b->bb[1];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=NATK[sq]&empty;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Bishops — non-capture */
        { uint64_t tmp=b->bb[2];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=bish_attacks(sq,occ)&empty;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Rooks — non-capture */
        { uint64_t tmp=b->bb[3];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=rook_attacks(sq,occ)&empty;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Queens — non-capture */
        { uint64_t tmp=b->bb[4];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=queen_attacks(sq,occ)&empty;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* King — non-capture + castles */
        { int sq=b->wk;
          uint64_t atk=KATK[sq]&empty;
          while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          if(sq==60){
            if((b->ca&CA_WK)&&bd[63]==WR&&!bd[61]&&!bd[62]&&
               !board_is_attacked(b,60,COL_B)&&!board_is_attacked(b,61,COL_B)&&!board_is_attacked(b,62,COL_B))
              PUSH(60,62,0,0,1);
            if((b->ca&CA_WQ)&&bd[56]==WR&&!bd[59]&&!bd[58]&&!bd[57]&&
               !board_is_attacked(b,60,COL_B)&&!board_is_attacked(b,59,COL_B)&&!board_is_attacked(b,58,COL_B))
              PUSH(60,58,0,0,2);
          }
        }
    } else {
        /* Black pawn pushes (non-promotion) */
        { uint64_t tmp=b->bb[6];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            int r=sq>>3,f=sq&7;
            int s1=sq+8;
            if(s1<64&&!bd[s1]&&r!=6){
                PUSH(sq,s1,0,0,0);
                if(r==1&&!bd[sq+16]) PUSH(sq,sq+16,0,0,0);
            }
          }
        }
        /* Knights */
        { uint64_t tmp=b->bb[7];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=NATK[sq]&empty;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Bishops */
        { uint64_t tmp=b->bb[8];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=bish_attacks(sq,occ)&empty;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Rooks */
        { uint64_t tmp=b->bb[9];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=rook_attacks(sq,occ)&empty;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* Queens */
        { uint64_t tmp=b->bb[10];
          while(tmp){int sq=lsb64(tmp);tmp&=tmp-1;
            uint64_t atk=queen_attacks(sq,occ)&empty;
            while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          }
        }
        /* King + castles */
        { int sq=b->bk;
          uint64_t atk=KATK[sq]&empty;
          while(atk){PUSH(sq,lsb64(atk),0,0,0);atk&=atk-1;}
          if(sq==4){
            if((b->ca&CA_BK)&&bd[7]==BR&&!bd[5]&&!bd[6]&&
               !board_is_attacked(b,4,COL_W)&&!board_is_attacked(b,5,COL_W)&&!board_is_attacked(b,6,COL_W))
              PUSH(4,6,0,0,3);
            if((b->ca&CA_BQ)&&bd[0]==BR&&!bd[3]&&!bd[2]&&!bd[1]&&
               !board_is_attacked(b,4,COL_W)&&!board_is_attacked(b,3,COL_W)&&!board_is_attacked(b,2,COL_W))
              PUSH(4,2,0,0,4);
          }
        }
    }
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
 * MAKE / UNMAKE
 * ═══════════════════════════════════════════════════════════════════ */
static const int CASTLE_SQ[5][4]={{0,0,0,0},{60,62,63,61},{60,58,56,59},{4,6,7,5},{4,2,0,3}};

/* ── Diff-based bb recording helpers ─────────────────────────────────────────
 * Each bb op is packed as: (bi<<7)|(sq<<1)|is_set  in a uint16_t.
 * Undo: if is_set, we set that bit again → means the make *cleared* it, so to
 * undo we need to SET it back.  Conversely if !is_set, we clear it.
 * In other words we record what the bit WAS before the make; restore by writing it back.
 * Encoding: bit[0]=old_value, bits[1..6]=sq, bits[7..10]=bi.                    */
#define UF_BB_ENCODE(bi,sq,was_set) ((uint16_t)(((bi)<<7)|((sq)<<1)|(was_set)))
#define UF_BB_BI(op)   ((op)>>7)
#define UF_BB_SQ(op)   (((op)>>1)&63)
#define UF_BB_WAS(op)  ((op)&1)

/* Record a bb change: old value of bit sq in bb[bi] before clr/set */
static inline void uf_bb_record(UndoFrame *uf, int bi, int sq, int was_set) {
    uf->bb_ops[uf->nbb++] = UF_BB_ENCODE(bi,sq,was_set);
}

/* Wrappers that record + apply */
static inline void uf_bb_clr(Board *b, UndoFrame *uf, int bi, int sq) {
    int was = !!(b->bb[bi] & ((uint64_t)1<<sq));
    uf_bb_record(uf, bi, sq, was);
    b->bb[bi] &= ~((uint64_t)1<<sq);
}
static inline void uf_bb_set(Board *b, UndoFrame *uf, int bi, int sq) {
    int was = !!(b->bb[bi] & ((uint64_t)1<<sq));
    uf_bb_record(uf, bi, sq, was);
    b->bb[bi] |= ((uint64_t)1<<sq);
}

/* Record a board square change and update */
static inline void uf_sq_set(Board *b, UndoFrame *uf, int sq, uint8_t new_pc) {
    uf->sq[uf->nsq]  = (uint8_t)sq;
    uf->pc[uf->nsq]  = b->b[sq];   /* save OLD value */
    uf->nsq++;
    b->b[sq] = new_pc;
}

void board_make(Board *b, const Move *m) {
    if(*b->undo_top>=STACK_SIZE){fprintf(stderr,"[BUG] undo overflow\n");return;}
    UndoFrame *uf=&b->undo[(*b->undo_top)++];

    /* Save scalars */
    uf->hash=b->hash; uf->turn=b->turn; uf->ca=b->ca; uf->ep=b->ep;
    uf->hm=b->hm; uf->fm=b->fm;
    uf->castled_w=b->castled_w; uf->castled_b=b->castled_b;
    uf->wk=b->wk; uf->bk=b->bk;
    uf->occ=b->occ; uf->occ_w=b->occ_w; uf->occ_b=b->occ_b;
    uf->nsq=0; uf->nbb=0;

    if(b->hist_len<HIST_SIZE){b->hist[b->hist_len]=b->hash;}
    b->hist_len++;

    int f=m->from,to=m->to;
    uint8_t p=b->b[f],cap=b->b[to];
    int col=b->turn;
    /* ═══════════════════════════════════════════════════════════════
     * NNUE ACCUMULATOR PUSH  (v3.13 — CRITICAL, DO NOT REMOVE)
     * ═══════════════════════════════════════════════════════════════
     *
     * HOW THE ACCUMULATOR STACK WORKS:
     * nnue_push() increments global _acc_ptr and writes an incrementally-
     * updated HM accumulator at _acc_buf_w/b[_acc_ptr] based on the
     * move delta (added/removed pieces).  nnue_eval_bb() then reads
     * from _acc_buf_w/b[_acc_ptr] to get the correct HM features for
     * the CURRENT position.  nnue_pop() (in board_unmake) decrements
     * _acc_ptr, restoring the parent's accumulator.
     *
     * The stack grows with search depth: root=0, after 1 make=1, etc.
     * Max depth is bounded by MAX_PLY (128), matching the stack size.
     *
     * WHY THIS IS ESSENTIAL:
     * Without push/pop, _acc_ptr stays at 0 (root position).  The HM
     * features (768 king-piece buckets) NEVER change, so nnue_eval_bb
     * returns the SAME base score for every position.  Only the 31 ext
     * features (piece counts, passed pawns, king distance) vary, but
     * they're too weak to differentiate positions.  Result: every node
     * evaluates to ±42cp and the engine plays a2a3, a3a4 (random moves
     * that happen to have lowest move index).
     *
     * VERIFIED: Removing these 4 lines reproduces the ±42cp bug.
     * Keeping them produces bit-identical results to v3.12.
     *
     * THREAD SAFETY CAVEAT:
     * _acc_ptr and _acc_buf are GLOBAL — multiple Lazy SMP threads
     * write them concurrently.  This is a benign data race: each
     * thread may read a stale accumulator, but since Lazy SMP threads
     * are independent searches, the impact is limited to slightly
     * inaccurate evals (not crashes).
     * TODO: Move _acc_ptr/_acc_buf into NnueAccum for true safety. */
    if(nnue_ready()){
        NNMove nm; nm.from_sq=(uint8_t)f; nm.to_sq=(uint8_t)to;
        nm.prom=m->prom; nm.is_epc=m->epc; nm.castle=m->castle;
        nnue_push(b->b,&nm);
    }

    if(m->castle){
        int kf=CASTLE_SQ[m->castle][0],kt=CASTLE_SQ[m->castle][1];
        int rf=CASTLE_SQ[m->castle][2],rt=CASTLE_SQ[m->castle][3];
        uint8_t kp=b->b[kf],rp=b->b[rf];
        b->hash^=ZR_tab[kp*64+kf];
        b->hash^=ZR_tab[rp*64+rf];
        uf_bb_clr(b,uf,P2BI[kp],kf); uf_bb_clr(b,uf,P2BI[rp],rf);
        uf_sq_set(b,uf,kt,kp); uf_sq_set(b,uf,rt,rp);
        uf_sq_set(b,uf,kf,0);  uf_sq_set(b,uf,rf,0);
        uf_bb_set(b,uf,P2BI[kp],kt); uf_bb_set(b,uf,P2BI[rp],rt);
        b->hash^=ZR_tab[kp*64+kt];
        b->hash^=ZR_tab[rp*64+rt];
        if(col==COL_W){b->castled_w=1;b->wk=(uint8_t)kt;}
        else           {b->castled_b=1;b->bk=(uint8_t)kt;}
    } else {
        if(cap){b->hash^=ZR_tab[cap*64+to]; uf_bb_clr(b,uf,P2BI[cap],to);}
        if(m->epc){
            int epsq=col==COL_W?to+8:to-8;
            uint8_t ep_p=b->b[epsq];
            b->hash^=ZR_tab[ep_p*64+epsq];
            uf_bb_clr(b,uf,P2BI[ep_p],epsq);
            uf_sq_set(b,uf,epsq,0);
        }
        b->hash^=ZR_tab[p*64+f];
        uf_bb_clr(b,uf,P2BI[p],f);
        uint8_t np=m->prom?(uint8_t)(col|m->prom):p;
        uf_sq_set(b,uf,to,np); uf_sq_set(b,uf,f,0);
        uf_bb_set(b,uf,P2BI[np],to);
        b->hash^=ZR_tab[np*64+to];
        if(PC_TYPE(p)==6){if(col==COL_W)b->wk=(uint8_t)to;else b->bk=(uint8_t)to;}
    }
    uint8_t old_ca=b->ca;
    if(PC_TYPE(p)==6){if(col==COL_W)b->ca&=~(CA_WK|CA_WQ);else b->ca&=~(CA_BK|CA_BQ);}
    /* Standard squares — always clear when touched (covers normal chess and
     * Fischer Random positions where the rook happens to sit on a corner). */
    if(f==63||to==63)b->ca&=~CA_WK;if(f==56||to==56)b->ca&=~CA_WQ;
    if(f==7 ||to==7 )b->ca&=~CA_BK;if(f==0 ||to==0 )b->ca&=~CA_BQ;
    if(PC_TYPE(p)==4){   /* rook */
        if(col==COL_W){
            if(b->ca&CA_WK){
                if((f>>3)==7 && f>b->wk) b->ca&=~CA_WK;
            }
            if(b->ca&CA_WQ){
                if((f>>3)==7 && f<b->wk) b->ca&=~CA_WQ;
            }
        } else {
            if(b->ca&CA_BK){
                if((f>>3)==0 && f>b->bk) b->ca&=~CA_BK;
            }
            if(b->ca&CA_BQ){
                if((f>>3)==0 && f<b->bk) b->ca&=~CA_BQ;
            }
        }
    }
    /* Also revoke if a rook is CAPTURED on its original square (FRC). */
    if(cap && PC_TYPE(cap)==4){
        if(PC_COLOR(cap)==COL_W){
            if((b->ca&CA_WK) && (to>>3)==7 && to>b->wk) b->ca&=~CA_WK;
            if((b->ca&CA_WQ) && (to>>3)==7 && to<b->wk) b->ca&=~CA_WQ;
        } else {
            if((b->ca&CA_BK) && (to>>3)==0 && to>b->bk) b->ca&=~CA_BK;
            if((b->ca&CA_BQ) && (to>>3)==0 && to<b->bk) b->ca&=~CA_BQ;
        }
    }
    if(old_ca!=b->ca){b->hash^=ZR_castle[old_ca]^ZR_castle[b->ca];}
    int8_t old_ep=b->ep;
    int df=to-f; if(df<0)df=-df;
    b->ep=(PC_TYPE(p)==1&&df==16)?(int8_t)((f+to)>>1):-1;
    if(old_ep>=0){b->hash^=ZR_ep[old_ep&7];}
    if(b->ep >=0){b->hash^=ZR_ep[b->ep &7];}
    b->hm=(PC_TYPE(p)==1||cap||m->epc)?0:b->hm+1;
    if(col==COL_B)b->fm++;
    b->hash^=ZR_side;
    b->turn=col^24;
    /* Incremental occupancy update (Phase 4 v212B):
     * Instead of rebuilding from 12 ORs, compute occ from the old saved value.
     * The bb[] array has already been updated, so we can derive occ cheaply
     * by XOR-ing the from/to/capture bits. But it's even safer and simpler
     * to just compute from the 6 bb per side (only 5+5 ORs vs 12). */
    b->occ_w = b->bb[0]|b->bb[1]|b->bb[2]|b->bb[3]|b->bb[4]|b->bb[5];
    b->occ_b = b->bb[6]|b->bb[7]|b->bb[8]|b->bb[9]|b->bb[10]|b->bb[11];
    b->occ   = b->occ_w | b->occ_b;
}

void board_unmake(Board *b) {
    if(nnue_ready())nnue_pop();  /* Undo incremental accumulator update (v3.13) */
    b->hist_len--;
    UndoFrame *uf=&b->undo[--(*b->undo_top)];

    /* Restore scalars */
    b->hash=uf->hash; b->turn=uf->turn; b->ca=uf->ca; b->ep=uf->ep;
    b->hm=uf->hm; b->fm=uf->fm;
    b->castled_w=uf->castled_w; b->castled_b=uf->castled_b;
    b->wk=uf->wk; b->bk=uf->bk;
    b->occ=uf->occ; b->occ_w=uf->occ_w; b->occ_b=uf->occ_b;

    /* Restore board squares (reverse order to handle overlapping writes correctly) */
    for (int i = uf->nsq-1; i >= 0; i--)
        b->b[uf->sq[i]] = uf->pc[i];

    /* Restore bitboards: undo each recorded op */
    for (int i = uf->nbb-1; i >= 0; i--) {
        uint16_t op = uf->bb_ops[i];
        int bi = UF_BB_BI(op), sq = UF_BB_SQ(op), was = UF_BB_WAS(op);
        if (was) b->bb[bi] |=  ((uint64_t)1<<sq);
        else     b->bb[bi] &= ~((uint64_t)1<<sq);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * DRAW DETECTION
 * ═══════════════════════════════════════════════════════════════════ */
int board_is_draw(const Board *b) {
    if(b->hm>=100)return 1;

    /* Insufficient material: positions where NO possible checkmate exists.
     * bb[] layout: 0=WP 1=WN 2=WB 3=WR 4=WQ 5=WK 6=BP 7=BN 8=BB 9=BR 10=BQ 11=BK
     * If no pawns, rooks, or queens exist, check for KvK / KNvK / KBvK. */
    uint64_t non_king_major = b->bb[0] | b->bb[6]    /* pawns   */
                            | b->bb[3] | b->bb[9]    /* rooks   */
                            | b->bb[4] | b->bb[10];  /* queens  */
    if (!non_king_major) {
        uint64_t minors = b->bb[1] | b->bb[7]   /* knights */
                        | b->bb[2] | b->bb[8];  /* bishops */
        if (__builtin_popcountll(minors) <= 1) return 1;  /* KvK, KNvK, KBvK */
    }

    if(b->hm<4||b->hist_len<2)return 0;
    uint64_t h=b->hash;
    int reps=1;
    int limit=b->hist_len-(b->hm+2<100?b->hm+2:100);
    if(limit<0)limit=0;
    for(int i=b->hist_len-2;i>=limit;i-=2)
        if(b->hist[i]==h){if(++reps>=3)return 2;}
    if(reps==2)return 3;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * UCI MOVE HELPERS
 * ═══════════════════════════════════════════════════════════════════ */
void move_to_uci(const Move *m, char *buf) {
    static const char *FILES="abcdefgh",*RANKS="87654321";
    static const char PROM_CHAR[6]={0,0,'n','b','r','q'};
    if(!m->from&&!m->to){strcpy(buf,"0000");return;}
    buf[0]=FILES[m->from&7]; buf[1]=RANKS[m->from>>3];
    buf[2]=FILES[m->to&7];   buf[3]=RANKS[m->to>>3];
    if(m->prom){buf[4]=PROM_CHAR[m->prom];buf[5]=0;}else{buf[4]=0;}
}

int move_from_uci(Board *b, const char *uci, Move *out) {
    static const char *FILES="abcdefgh",*RANKS="87654321";
    if(!uci||strlen(uci)<4)return 0;
    const char *fc=strchr(FILES,uci[0]),*fr=strchr(RANKS,uci[1]);
    const char *tc=strchr(FILES,uci[2]),*tr=strchr(RANKS,uci[3]);
    if(!fc||!fr||!tc||!tr)return 0;
    int from=(int)(fr-RANKS)*8+(int)(fc-FILES);
    int to  =(int)(tr-RANKS)*8+(int)(tc-FILES);
    uint8_t prom=0;
    if(uci[4]){char pc=uci[4];if(pc=='n'||pc=='N')prom=2;else if(pc=='b'||pc=='B')prom=3;else if(pc=='r'||pc=='R')prom=4;else if(pc=='q'||pc=='Q')prom=5;}
    Move moves[MAX_MOVES]; int n=board_gen_moves(b,moves);
    for(int i=0;i<n;i++)if(moves[i].from==from&&moves[i].to==to&&moves[i].prom==prom){*out=moves[i];return 1;}
    return 0;
}
