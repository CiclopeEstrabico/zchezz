/* board.h — Zchezz board state, move generation, make/unmake
 *
 * Square layout: 0=a8, 7=h8, 56=a1, 63=h1
 * Bitboards: uint64_t bb[12], one per piece type
 *   bb[0]=WP bb[1]=WN bb[2]=WB bb[3]=WR bb[4]=WQ bb[5]=WK
 *   bb[6]=BP bb[7]=BN bb[8]=BB bb[9]=BR bb[10]=BQ bb[11]=BK
 */
#pragma once
#include <stdint.h>

#define COL_W   8
#define COL_B  16
#define WP  9
#define WN 10
#define WB 11
#define WR 12
#define WQ 13
#define WK 14
#define BP 17
#define BN 18
#define BB 19
#define BR 20
#define BQ 21
#define BK 22

#define PC_COLOR(p)  ((p) & 24)
#define PC_TYPE(p)   ((p) &  7)

#define CA_WK  1
#define CA_WQ  2
#define CA_BK  4
#define CA_BQ  8

#define MAX_MOVES 256

typedef struct {
    uint8_t  from;
    uint8_t  to;
    uint8_t  prom;
    uint8_t  epc;
    uint8_t  castle;
    int32_t  score;
} Move;

#define STACK_SIZE 512
#define HIST_SIZE  1024

typedef struct {
    uint8_t  b[64];
    uint64_t bb[12];
    uint64_t occ;     /* all pieces occupancy (cached) */
    uint64_t occ_w;   /* white pieces occupancy        */
    uint64_t occ_b;   /* black pieces occupancy        */
    int      turn;
    uint8_t  ca;
    int8_t   ep;
    uint8_t  hm;
    uint16_t fm;
    uint8_t  wk;
    uint8_t  bk;
    int      castled_w;
    int      castled_b;
    uint64_t hash;
    uint64_t hist [HIST_SIZE];
    int      hist_len;
} Board;

/* Diff-based undo frame — stores only the squares that changed (max 4 for castles)
 * and the scalar fields, eliminating 160 bytes of memcpy per make/unmake.       */
typedef struct {
    /* Scalar state (restored directly) */
    uint64_t hash;
    uint64_t occ, occ_w, occ_b;   /* cached occupancy (Phase 1) */
    int      turn;
    uint8_t  ca;
    int8_t   ep;
    uint8_t  hm;
    uint16_t fm;
    uint8_t  wk, bk;
    int      castled_w, castled_b;

    /* Square diff: up to 4 squares touched (kf, kt, rf, rt for castles;
     * from, to, ep_sq for normal moves — at most 3 squares, padded to 4) */
    uint8_t  sq[4];    /* square indices                                */
    uint8_t  pc[4];    /* old piece at that square (0 = empty)          */
    uint8_t  nsq;      /* number of valid entries (1..4)                */

    /* Bitboard diff: one bb entry per changed square (same order as sq[]) */
    /* We record old bb state by inverting the make operations, so we just
     * store the pre-move bb entries for the affected indices.             */
    /* Actually we record the 4 (bi, sq, was_set) tuples to undo bb ops. */
    /* Each op: bi (4 bits) | sq (6 bits) | set (1 bit) = 11 bits → pack in uint16_t */
    uint16_t bb_ops[8];   /* up to 8 bitboard operations to undo         */
    uint8_t  nbb;          /* number of bb ops                            */
} UndoFrame;

extern UndoFrame g_undo[STACK_SIZE];
extern int       g_undo_top;
extern uint64_t  g_hist [HIST_SIZE];
extern int       g_hist_len;

extern uint64_t ZR_tab   [32*64];
extern uint64_t ZR_ep    [8];
extern uint64_t ZR_castle[16];
extern uint64_t ZR_side;

/* Leaper attack tables */
extern uint64_t NATK[64];
extern uint64_t KATK[64];
extern int8_t   NATK_ARR[64][8];
extern int8_t   KATK_ARR[64][8];
extern int      NATK_N  [64];
extern int      KATK_N  [64];
extern uint8_t  DIST[64*64];

/* Magic bitboard tables */
extern uint64_t ROOK_MAGIC[64];
extern uint64_t ROOK_MASK [64];
extern int      ROOK_SHIFT[64];
extern uint64_t *ROOK_TBL [64];   /* pointer into shared attack table */

extern uint64_t BISH_MAGIC[64];
extern uint64_t BISH_MASK [64];
extern int      BISH_SHIFT[64];
extern uint64_t *BISH_TBL [64];

extern int MV_TAB[7];

void    board_init(void);
int     board_load_fen(Board *b, const char *fen);
void    board_to_fen(const Board *b, char *buf);
void    board_make  (Board *b, const Move *m);
void    board_unmake(Board *b);
int     board_gen_moves   (const Board *b, Move *out);
int     board_gen_captures(const Board *b, Move *out);
int     board_is_attacked(const Board *b, int sq, int by);
int     board_in_check(const Board *b);
uint64_t board_compute_hash(const uint8_t *brd, int turn, uint8_t ca, int ep);
int     board_is_draw(const Board *b);
void    move_to_uci(const Move *m, char *buf);
int     move_from_uci(Board *b, const char *uci, Move *out);

/* Legal move generation helpers (Phase 5 v212B) */
uint64_t board_checkers(const Board *b);   /* pieces giving check to stm king */
uint64_t board_pinned  (const Board *b);   /* stm pieces pinned to stm king  */
/* Pin ray: given pinned square, returns the ray mask (king..pinner inclusive) */
uint64_t board_pin_ray (const Board *b, int pinned_sq);

static inline uint64_t rook_attacks(int sq, uint64_t occ) {
    return ROOK_TBL[sq][((occ & ROOK_MASK[sq]) * ROOK_MAGIC[sq]) >> ROOK_SHIFT[sq]];
}
static inline uint64_t bish_attacks(int sq, uint64_t occ) {
    return BISH_TBL[sq][((occ & BISH_MASK[sq]) * BISH_MAGIC[sq]) >> BISH_SHIFT[sq]];
}
static inline uint64_t queen_attacks(int sq, uint64_t occ) {
    return rook_attacks(sq,occ) | bish_attacks(sq,occ);
}

/* ── Pawn attack bitboards (Phase 2) ──────────────────────────── */
#define FILE_A 0x0101010101010101ULL
#define FILE_H 0x8080808080808080ULL

/* Bulk pawn attacks: all squares attacked by a set of pawns */
static inline uint64_t wpawn_attacks_bb(uint64_t pawns) {
    return ((pawns >> 7) & ~FILE_A) | ((pawns >> 9) & ~FILE_H);
}
static inline uint64_t bpawn_attacks_bb(uint64_t pawns) {
    return ((pawns << 7) & ~FILE_H) | ((pawns << 9) & ~FILE_A);
}
