/* nnue.h — Zchezz NNUE evaluation layer (v3.13: thread-safe accumulators)
 *
 * Architecture : 768 → 256 → 64 → 1   (int16/int8/float32)
 * Weight file  : NNU3 binary (quantized)
 * Perspective  : dual-accumulator (White-POV + Black-POV)
 * Output       : centipawns, White-relative, clamped to [-2000, +2000]
 *                cp = logit * 320.0f
 *
 * ── Thread safety model (v3.13) ──────────────────────────────────
 *
 * PROBLEM (v3.12 and earlier):
 *   The NNUE accumulator buffers (_acc_buf_w, _acc_buf_b, _ext_buf,
 *   _ext_cache_*) were global statics.  When Lazy SMP helper threads
 *   call nnue_eval concurrently, they corrupt each other's accumulator
 *   state, producing garbage evaluations.  Result: 4T scored 0.5/50
 *   against 1T.
 *
 * SOLUTION (v3.13):
 *   All per-evaluation mutable state is collected into a small struct
 *   (NnueAccum, ~4.3 KB).  A pointer to this struct lives in Board,
 *   following the same pattern used for the undo stack in v3.12:
 *     - Main thread  → static global NnueAccum (zero allocation)
 *     - SMP helpers  → heap-allocated NnueAccum per thread
 *   Cost: one extra pointer dereference per nnue_eval call — same as
 *   the undo pointer, which was proven to have zero NPS impact.
 *
 * WHY NOT Thread-Local Storage (TLS):
 *   MinGW on Windows uses __emutls_get_address, which inserts a
 *   function call on every TLS access.  This caused ~15% NPS loss
 *   in v3.11.  The pointer-in-Board approach avoids TLS entirely.
 *
 * WHAT REMAINS GLOBAL (read-only, safe for MT):
 *   All weight matrices (_nnL1WT, _nnL2W, _nnL3W, biases) are loaded
 *   once at startup and never modified.  They are safely shared.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

/* ── Layer dimensions ──────────────────────────────────────────── */
/* Input = 768 half-mirror (HM) features + 31 endgame features = 799 */
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
/* board_make/unmake call nnue_push/nnue_pop to incrementally update
 * the global accumulator stack (_acc_buf_w/b[_acc_ptr]).  nnue_eval_bb
 * reads the accumulator from these globals, not from NnueAccum.
 * NnueAccum only holds the ext feature cache (per-thread for SMP). */
#define NN_ACC_DEPTH 512

/* ── Extra-feature cache slots ─────────────────────────────────── */
/* Small direct-mapped cache to avoid recomputing the 31 endgame
 * feature projections when the same position is evaluated from
 * both sides (White-POV then Black-POV) or revisited.
 * 4 slots × 512 bytes = 2 KB overhead per thread. */
#define EXT_CACHE_SLOTS 4

/* ── Per-thread NNUE accumulator state (v3.13) ─────────────────
 *
 * Contains all mutable state needed for nnue_eval / nnue_eval_bb.
 * Each search thread (main + Lazy SMP helpers) owns one instance.
 *
 * Memory layout (~4.3 KB, 32-byte aligned for AVX2 SIMD):
 *   acc_w[256]         — White-perspective accumulator (512 B)
 *   acc_b[256]         — Black-perspective accumulator (512 B)
 *   ext_buf[2][256]    — extra-feature projection buffers (1 KB)
 *   ext_feat[2][31]    — raw extra-feature values (248 B)
 *   ext_dirty[2]       — dirty flags per perspective (2 B)
 *   acc_dirty           — rebuild-needed flag (4 B)
 *   cache_key[4]       — ext cache hash keys (32 B)
 *   cache_buf[4][256]  — ext cache projected buffers (2 KB)
 *
 * Allocation strategy:
 *   Main thread  → static global `g_nnue_accum` (declared in board.h)
 *   SMP helpers  → heap-allocated in helper_thread_fn (main.c)
 *   WASM         → static global (single-threaded) */
typedef struct {
    int16_t  acc_w[NN_L1_OUT]                __attribute__((aligned(32)));
    int16_t  acc_b[NN_L1_OUT]                __attribute__((aligned(32)));
    int16_t  ext_buf[2][NN_L1_OUT]           __attribute__((aligned(32)));
    float    ext_feat[2][NN_EXTRA];
    int8_t   ext_dirty[2];
    int      acc_dirty;
    uint64_t cache_key[EXT_CACHE_SLOTS];
    int16_t  cache_buf[EXT_CACHE_SLOTS][NN_L1_OUT] __attribute__((aligned(32)));
} NnueAccum;

/* ── Public API ────────────────────────────────────────────────── */

/* Load NNU3 binary from path.  Returns 0 on success, -1 on error.
 * Thread safety: call once from main thread before any search. */
int  nnue_load(const char *path);

/* True after a successful nnue_load(). Thread-safe (read-only flag). */
int  nnue_ready(void);

/* Fully rebuild the accumulator from scratch.
 * Writes acc_w[] and acc_b[] in `na` by scanning all 768 HM features
 * from board[64].  Called once at the start of each search iteration.
 *
 * na:       per-thread accumulator (Board::nnue)
 * board[64]: piece constants (WP=9..BK=22, empty=0) */
void nnue_rebuild(NnueAccum *na, const uint8_t *board);

/* NNMove — describes a move for incremental push (legacy API).
 * Not used by search (which does rebuild per iteration), but kept
 * for NNUE test functions and potential future incremental updates. */
typedef struct {
    uint8_t from_sq;
    uint8_t to_sq;
    uint8_t prom;       /* 0 = not a promotion */
    uint8_t is_epc;     /* en-passant capture flag */
    uint8_t castle;     /* 0|1|2|3|4 */
} NNMove;

/* Incremental accumulator update — push/pop on the global array stack.
 * Called by board_make/board_unmake.  nnue_eval_bb reads from these
 * global arrays for the HM accumulator (not from NnueAccum). */
void nnue_push(const uint8_t *board, const NNMove *m);
void nnue_pop(void);

/* Reset accumulator state.  Marks `na` as dirty (needs rebuild)
 * and clears the extra-feature cache.
 * Called at start of every search (search_best). */
void nnue_reset(NnueAccum *na);

/* Forward pass (basic).  Computes the 31 endgame features from
 * board[64] on every call.
 *
 * na:       per-thread accumulator
 * stm:      0 = White to move, 1 = Black
 * board[64]: current position
 * Returns:  centipawns, White-relative, clamped [-2000, +2000] */
int  nnue_eval(NnueAccum *na, int stm, const uint8_t *board);

/* Fast path for search.  Accepts precomputed bb[12] bitboards
 * from Board and the Zobrist hash for a per-thread ext cache.
 * Avoids the 64-square _build_bb_from_board scan on every call.
 *
 * na:         per-thread accumulator
 * stm:        0 = White, 1 = Black
 * board[64]:  current mailbox
 * bb[12]:     precomputed bitboards (Board::bb)
 * board_hash: Zobrist hash (Board::hash), used as cache key */
int  nnue_eval_bb(NnueAccum *na, int stm, const uint8_t *board,
                  const uint64_t bb[12], uint64_t board_hash);

/* Load weights from a memory buffer (for WebAssembly / browser use).
 * data must point to the raw NNU3 binary content, len is the byte count.
 * Returns 0 on success, -1 on error.  Thread safety: call once. */
int nnue_load_from_mem(const uint8_t *data, size_t len);

/* WASM backward-compatibility wrapper.
 * JS worker calls nnue_reset() with no args — this passes the global
 * accumulator.  For native builds, use nnue_reset(na) directly. */
void nnue_reset_global(void);
