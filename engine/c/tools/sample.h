/* sample.h — the packed .bin training-sample record, shared by every
 * tool that produces training data (tools/selfplay.c, tools/arena.c).
 *
 * THIS STRUCT IS A CROSS-LANGUAGE CONTRACT with train/dataset.py's
 * SAMPLE_DTYPE. It lives in its own header precisely so that two
 * producers cannot drift apart: a second copy-pasted definition would
 * compile fine and silently write a differently-laid-out file that the
 * Python reader would reinterpret as garbage. The _Static_assert below
 * is the only thing standing between a field reorder and a training run
 * on noise — keep it, and update dataset.py in the same commit as any
 * change here.
 *
 * Field semantics (see selfplay.c's header for the full rationale):
 *   board[64]    mailbox, Zchezz encoding (0=empty, WP=9..BK=22), sq 0 = a8
 *   stm          0 = white to move, 1 = black
 *   rule50       halfmove clock
 *   castling     bitmask
 *   ep_file      0..7, 8 = none
 *   eval_cp      STM-relative score from the search that CHOSE the move
 *   game_result  +1/0/-1 from the point of view of the side to move in
 *                THIS position (requires a second pass once the game ends)
 *   move_played  packed move, low 16 bits of search.c's pack_move layout

 * ═══════════════════════════════════════════════════════════════════
 *  LABEL CONVENTION (CLAUDE.md rule 10) — result / cp / wdl
 * ═══════════════════════════════════════════════════════════════════
 *
 *  | Name    | Meaning                          | Range / frame        |
 *  |---------|----------------------------------|----------------------|
 *  | result  | the REAL game outcome            | parquet: 0.0/0.5/1.0 |
 *  |         |                                  | WHITE-relative       |
 *  | cp      | evaluation in centipawns         | int                  |
 *  | wdl     | sigmoid(cp/320), a FUNCTION of   | 0..1                 |
 *  |         | cp — NOT an outcome              |                      |
 *  | target  | lam*result + (1-lam)*wdl         | computed at TRAINING |
 *  |         |                                  | time, never stored   |
 *
 *  THIS FILE'S .bin RECORD USES ITS OWN INTERNALLY-CONSISTENT FRAME:
 *  `eval_cp` and `game_result` (+1/0/-1) are BOTH STM-relative here.
 *  train/dataset.py converts game_result to the 0..1 probability
 *  ((g+1)/2) at read time. Do not mix the two frames.
 *
 *  Never bake the lam blend into a stored record — lam is per-dataset and
 *  is annealed across bootstrap generations.
 * ═══════════════════════════════════════════════════════════════════
 */
#pragma once
#include <stdint.h>

#pragma pack(push, 1)
typedef struct {
    uint8_t  board[64];
    uint8_t  stm;
    uint8_t  rule50;
    uint8_t  castling;
    uint8_t  ep_file;
    int16_t  eval_cp;
    int8_t   game_result;
    uint16_t move_played;
    uint16_t _pad;
} SelfplaySample;
#pragma pack(pop)

#define SELFPLAY_SAMPLE_SIZE 75
/* Compile-time contract check (C11 _Static_assert) — if this ever
 * fires, the struct drifted from dataset.py's SAMPLE_DTYPE and
 * every downstream .bin read would silently reinterpret garbage. */
_Static_assert(sizeof(SelfplaySample) == SELFPLAY_SAMPLE_SIZE,
    "SelfplaySample size drifted from train/dataset.py's "
    "SAMPLE_DTYPE (75 bytes) — update both in lockstep.");


/* Pack a move into SAMPLE_DTYPE's 16-bit move_played field.
 *
 * Takes plain ints rather than a `Move` so this header stays free of
 * board.h — sample.h is included by tools that already include board.h
 * and (potentially) by tests that do not.
 *
 * This is the LOW 16 BITS of search.c's 20-bit pack_move layout:
 * from[0:5] to[6:11] prom[12:14] epc[15]. The 4 castle bits that
 * search.c's TT packing carries are dropped — not a loss for consumers,
 * since a castling move is always the king moving two files on its home
 * rank and from/to identify it unambiguously. If exact parity with the
 * 20-bit layout is ever needed, widen move_played to uint32 in BOTH this
 * header and dataset.py rather than silently reusing these bits. */
static inline uint16_t sample_pack_move(int from, int to, int prom, int epc) {
    return (uint16_t)((from & 63) | ((to & 63) << 6) |
                      ((prom & 7) << 12) | (epc ? (1 << 15) : 0));
}
