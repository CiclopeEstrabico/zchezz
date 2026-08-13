/* syzygy.c — Zchezz ↔ Fathom Syzygy integration
 *
 * Maps Zchezz Board bitboards to Fathom's format and provides
 * WDL/DTZ probing functions.
 *
 * Zchezz square mapping: a8=0  h8=7  a1=56 h1=63  (rank 8 first)
 * Fathom square mapping:  a1=0  h1=7  a8=56 h8=63  (rank 1 first)
 *
 * Bitboard conversion: vertical flip = __builtin_bswap64()
 * Square conversion:   fathom_sq = zchezz_sq ^ 56
 */

#ifndef NO_TABLEBASES

#include "syzygy.h"
#include "tbprobe.h"
#include <string.h>
#include <stdio.h>

/* ── Bitboard vertical flip ──────────────────────────────────── */
/* Reverses byte order = flips ranks (row 0 ↔ row 7) */
static inline uint64_t flip_bb(uint64_t bb) {
    return __builtin_bswap64(bb);
}

/* ── Square conversion ───────────────────────────────────────── */
/* Flip rank: row 0 (rank 8) → row 7 (rank 8 in Fathom), etc. */
static inline unsigned zchezz_to_fathom_sq(int zsq) {
    return (unsigned)(zsq ^ 56);
}

/* ── Init / Free ─────────────────────────────────────────────── */

int syzygy_init(const char *path) {
    if (!path || !path[0]) return 0;
    tb_init(path);
    return (int)TB_LARGEST;
}

void syzygy_free(void) {
    tb_free();
}

int syzygy_max_pieces(void) {
    return (int)TB_LARGEST;
}

/* ── Build Fathom bitboards from Board ───────────────────────── */
/* Fathom needs: white, black (all pieces), plus per-piece-type bitboards.
 * Zchezz bb[]: 0=WP 1=WN 2=WB 3=WR 4=WQ 5=WK 6=BP 7=BN 8=BB 9=BR 10=BQ 11=BK */

static void build_fathom_bbs(const Board *b,
                              uint64_t *white, uint64_t *black,
                              uint64_t *kings, uint64_t *queens,
                              uint64_t *rooks, uint64_t *bishops,
                              uint64_t *knights, uint64_t *pawns) {
    *white   = flip_bb(b->occ_w);
    *black   = flip_bb(b->occ_b);
    *pawns   = flip_bb(b->bb[0] | b->bb[6]);
    *knights = flip_bb(b->bb[1] | b->bb[7]);
    *bishops = flip_bb(b->bb[2] | b->bb[8]);
    *rooks   = flip_bb(b->bb[3] | b->bb[9]);
    *queens  = flip_bb(b->bb[4] | b->bb[10]);
    *kings   = flip_bb(b->bb[5] | b->bb[11]);
}

/* ── WDL Probe (for search) ──────────────────────────────────── */

int syzygy_probe_wdl(const Board *b, int *wdl) {
    /* Fathom requires no castling rights */
    if (b->ca != 0) return 0;

    /* Check piece count: skip KvK positions (2 pieces).
     * KvK is always a draw with no TB file. All 3-piece endgames
     * (KPK, KRK, KQK, KBK, KNK) have valid TB files and should
     * be probed — especially KPK which has complex win/draw boundaries.
     * Stockfish also probes 3-piece positions. */
    int npieces = __builtin_popcountll(b->occ);
    if (npieces < 3) return 0;
    if (npieces > (int)TB_LARGEST || TB_LARGEST == 0) return 0;

    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    build_fathom_bbs(b, &white, &black, &kings, &queens, &rooks, &bishops, &knights, &pawns);

    /* En passant square conversion */
    unsigned ep = 0;
    if (b->ep >= 0) {
        ep = zchezz_to_fathom_sq(b->ep);
    }

    /* Turn: true = white */
    int is_white = (b->turn == COL_W);

    { static long _dbgn=0; if ((++_dbgn % 200000)==0) { fprintf(stderr,"[dbg] probes=%ld\n", _dbgn); fflush(stderr);} }
    unsigned result = tb_probe_wdl(
        white, black, kings, queens, rooks, bishops, knights, pawns,
        (unsigned)b->hm,  /* rule50: Fathom rejects non-zero (only probe after cap/pawn) */
        0,                /* castling (already checked = 0) */
        ep,
        is_white
    );

    if (result == TB_RESULT_FAILED) return 0;

    *wdl = (int)result;
    return 1;
}

/* ── Root DTZ Probe ──────────────────────────────────────────── */

int syzygy_probe_root(const Board *b, int *best_from, int *best_to, int *wdl, int *dtz) {
    if (b->ca != 0) return 0;

    int npieces = __builtin_popcountll(b->occ);
    if (npieces < 3) return 0;
    if (npieces > (int)TB_LARGEST || TB_LARGEST == 0) return 0;

    uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    build_fathom_bbs(b, &white, &black, &kings, &queens, &rooks, &bishops, &knights, &pawns);

    unsigned ep = 0;
    if (b->ep >= 0) {
        ep = zchezz_to_fathom_sq(b->ep);
    }

    int is_white = (b->turn == COL_W);

    unsigned result = tb_probe_root(
        white, black, kings, queens, rooks, bishops, knights, pawns,
        (unsigned)b->hm,
        0,                /* castling */
        ep,
        is_white,
        NULL              /* no per-move results */
    );

    if (result == TB_RESULT_FAILED) return 0;

    /* Extract fields */
    *wdl = (int)TB_GET_WDL(result);
    *dtz = (int)TB_GET_DTZ(result);

    /* Convert move squares back to Zchezz notation */
    unsigned f_from = TB_GET_FROM(result);
    unsigned f_to   = TB_GET_TO(result);
    *best_from = (int)(f_from ^ 56);  /* Fathom → Zchezz */
    *best_to   = (int)(f_to   ^ 56);

    return 1;
}

#endif /* NO_TABLEBASES */
