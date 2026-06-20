/*
 * tbconfig.h — Zchezz adapter for Fathom tablebase probing library.
 *
 * This file configures Fathom to use its own internal move generator
 * (no dependency on Zchezz's board representation during probing).
 *
 * Zchezz square mapping: a8=0, h1=63 (rank 0 = rank 8)
 * Fathom square mapping: a1=0, h8=63 (standard)
 * Conversion: fathom_sq = zchezz_sq ^ 56
 */

#ifndef TBCONFIG_H
#define TBCONFIG_H

/* Use Fathom's own internal move generator.
 * This avoids coupling with the engine's board representation. */
#define TB_NO_HELPER_API

/* Thread safety: we call WDL from search (potentially multi-threaded later),
 * so do NOT define TB_NO_THREADS. Actually for now we're single-threaded
 * in search, but let's keep it safe. On WASM we don't use tablebases at all. */
/* #define TB_NO_THREADS */

/* King square is always available via bitboard scan — no special handling needed. */

/* Piece encoding: Fathom doesn't need this, it uses its own piece representation.
 * We only interact via the bitboard API (tb_probe_wdl, tb_probe_root). */

/* Value constants used by Fathom's DTM/DTZ root probing */
#ifndef TB_VALUE_INFINITE
#define TB_VALUE_INFINITE  32000
#endif
#ifndef TB_VALUE_MATE
#define TB_VALUE_MATE      30000
#endif
#ifndef TB_VALUE_PAWN
#define TB_VALUE_PAWN      100
#endif
#ifndef TB_VALUE_DRAW
#define TB_VALUE_DRAW      0
#endif
#ifndef TB_MAX_MATE_PLY
#define TB_MAX_MATE_PLY    256
#endif

typedef int Value;

#endif  /* TBCONFIG_H */
