/* syzygy.h — Zchezz ↔ Fathom Syzygy integration layer
 *
 * Provides a clean interface between Zchezz's Board and Fathom's tb_probe API.
 * Handles the square/bitboard remapping (Zchezz a8=0, Fathom a1=0).
 *
 * Compile with -DNO_TABLEBASES to stub everything out (WASM builds).
 */
#pragma once
#include <stdint.h>
#include "board.h"

#ifdef NO_TABLEBASES
/* ── Stubs for WASM / no-TB builds ──────────────────────────────── */
static inline int  syzygy_init(const char *path) { (void)path; return 0; }
static inline void syzygy_free(void) {}
static inline int  syzygy_max_pieces(void) { return 0; }
static inline int  syzygy_probe_wdl(const Board *b, int *wdl) { (void)b; (void)wdl; return 0; }
static inline int  syzygy_probe_root(const Board *b, int *best_from, int *best_to, int *wdl, int *dtz)
    { (void)b; (void)best_from; (void)best_to; (void)wdl; (void)dtz; return 0; }

#else
/* ── Full implementation ────────────────────────────────────────── */

/*
 * Initialize Syzygy tablebases.
 * path: semicolon-separated list of directories containing .rtbw/.rtbz files.
 * Returns: number of piece types loaded (0 if no files found, still success).
 */
int syzygy_init(const char *path);

/*
 * Free tablebase resources.
 */
void syzygy_free(void);

/*
 * Returns TB_LARGEST — the maximum number of pieces for which tables are loaded.
 */
int syzygy_max_pieces(void);

/*
 * Probe Win-Draw-Loss for position b (non-root, for use during search).
 * Requires: no castling, popcount(occ) <= syzygy_max_pieces(), rule50 == 0.
 * wdl output: 0=Loss, 1=BlessedLoss, 2=Draw, 3=CursedWin, 4=Win  (STM-relative)
 * Returns: 1 on success, 0 on failure (castling, too many pieces, etc.)
 */
int syzygy_probe_wdl(const Board *b, int *wdl);

/*
 * Probe Distance-To-Zero for root position.
 * Returns best move (from, to squares in Zchezz notation) and WDL/DTZ values.
 * Returns: 1 on success, 0 on failure.
 */
int syzygy_probe_root(const Board *b, int *best_from, int *best_to, int *wdl, int *dtz);

#endif /* NO_TABLEBASES */
