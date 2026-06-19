/* nnue.h — Zchezz NNUE evaluation layer
 *
 * Architecture : 768 → 256 → 64 → 1   (int16/int8/float32)
 * Weight file  : NNU3 binary (quantized)
 * Perspective  : dual-accumulator (White-POV + Black-POV), incremental updates
 * Output       : centipawns, White-relative, clamped to [-2000, +2000]
 *                cp = logit * 320.0f
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

/* ── Layer dimensions ──────────────────────────────────────────── */
/* Phase 1: input = 768 HM + 31 endgame features = 799             */
#define NN_L1_IN   799   /* 768 HM + 31 manual endgame features    */
#define NN_L1_OUT  256
#define NN_L2_IN   256
#define NN_L2_OUT   64   /* was 32 — doubled for Phase 1           */
#define NN_L3_IN    64   /* must equal NN_L2_OUT                   */
#define NN_L3_OUT    1

/* HM-only slice of the input (first 768 features).
 * The accumulator covers only these; the extra 31 endgame features
 * are appended in nnue_eval() each call from the live board state. */
#define NN_HM_IN   768   /* half-mirror encoding size (unchanged)  */
#define NN_EXTRA   31    /* manual endgame features appended after  */

/* ── Quantization constants ────────────────────────────────────── */
#define NN_QA      255
#define NN_QB      64
#define NN_SHIFT   8

/* ── Piece-type index (matches JS _NN_PT) ──────────────────────
 * P=0 N=1 B=2 R=3 Q=4 K=5                                        */
#define NN_NOTYPES  6
#define NN_NOCOLORS 2

/* ── Accumulator stack depth ───────────────────────────────────── */
#define NN_ACC_DEPTH 512

/* ── Public API ────────────────────────────────────────────────── */

/* Load NNU2 binary from path.  Returns 0 on success, -1 on error. */
int  nnue_load(const char *path);

/* True after a successful nnue_load(). */
int  nnue_ready(void);

/* Fully rebuild the accumulator at stack slot 0 from scratch.
 * board[64]: piece constants matching JS encoding
 *   White: WP=9 WN=10 WB=11 WR=12 WQ=13 WK=14
 *   Black: BP=17 BN=18 BB=19 BR=20 BQ=21 BK=22
 *   Empty: 0                                                       */
void nnue_rebuild(const uint8_t *board);

/* Push a new accumulator slot (copy + incremental update).
 * move fields: from_sq, to_sq, prom (0=none, 1-5=piece type),
 *              is_epc (en-passant capture), castle (0=none,
 *              1=K-side white, 2=Q-side white, 3=K-side black,
 *              4=Q-side black).
 * board[] is the position BEFORE the move.                         */
typedef struct {
    uint8_t from_sq;
    uint8_t to_sq;
    uint8_t prom;       /* 0 = not a promotion */
    uint8_t is_epc;     /* en-passant capture flag */
    uint8_t castle;     /* 0|1|2|3|4 */
} NNMove;

void nnue_push(const uint8_t *board, const NNMove *m);

/* Pop the top accumulator slot (called on unmake). */
void nnue_pop(void);

/* Reset accumulator pointer (called at start of every search). */
void nnue_reset(void);

/* Forward pass.  stm=0 for White to move, stm=1 for Black.
 * board[64]: the current board array (same encoding as nnue_rebuild).
 * Used to compute the 31 endgame features appended after the HM acc.
 * Returns centipawns, White-relative, clamped [-2000, +2000].      */
int  nnue_eval(int stm, const uint8_t *board);

/* Fast path for search: accepts precomputed bb[12] bitboards (from Board)
 * and the board's Zobrist hash for a 2-slot extra-feature cache.
 * Avoids the 64-iteration _build_bb_from_board scan on every eval call.
 * board_hash is used as a cache key; stm is XOR'd in to differentiate
 * the two side-to-move perspectives. */
int  nnue_eval_bb(int stm, const uint8_t *board,
                  const uint64_t bb[12], uint64_t board_hash);

/* Load weights from a memory buffer (for WebAssembly / browser use).
 * data must point to the raw NNU2 binary content, len is the byte count.
 * Returns 0 on success, -1 on error.                                     */
int nnue_load_from_mem(const uint8_t *data, size_t len);
