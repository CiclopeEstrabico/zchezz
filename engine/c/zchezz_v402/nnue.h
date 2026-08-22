/* nnue.h — Zchezz NNUE evaluation layer (v4.00: HalfKP-4Bucket, dual perspective)
 *
 * ── Architecture (v4.00) ─────────────────────────────────────────
 *
 *   Features  : HalfKP-4Bucket, 2560 per perspective
 *   L1        : 2560 → 512   (int16 accumulator, fully incremental)
 *   Act L1    : SCReLU  —  c = clamp(x, 0, QA); out = (c*c) >> 8
 *   Concat    : [L1(stm) 512 | L1(opp) 512] = 1024 uint8
 *   L2        : 1024 → 32    (int8, maddubs kernel)
 *   Act L2    : ClippedReLU [0, QB]
 *   L3        : 32   → 1     (int8 → float32)
 *   Output    : centipawns, STM-relative, clamped to [-2000, +2000]
 *   Weights   : NNU4 binary, ~2.6 MB
 *
 * ── What changed vs v3.14 (and why) ──────────────────────────────
 *
 * v3.14 used 768 half-mirror features (no king dependency) plus 31
 * hand-made "extra" endgame features (piece counts, passed pawns,
 * king distance).  Two problems:
 *
 *   1. NO KING DEPENDENCY.  The net could not tell "rook on e4 with
 *      king on g1" from "rook on e4 with king on a8".  This was the
 *      single biggest strength ceiling.
 *   2. THE 31 EXTRAS WERE NOT ACCUMULABLE.  They depend on several
 *      pieces at once, so every node paid _compute_extra_feat +
 *      _project_feat_full = O(31 x 256) SIMD that no cache could
 *      avoid, because the values change on every move.
 *
 * v4.00 fixes both: the king square selects one of 4 feature buckets
 * per perspective, and the extras are gone entirely — everything the
 * extras encoded emerges from the local features for free.  The
 * accumulator is now 100% incremental: the only non-incremental work
 * is a perspective rebuild when a king move crosses a bucket border
 * (see "Bucket refresh" below), which is rare.
 *
 * ── Coordinate and feature conventions (THE critical invariant) ──
 *
 * These MUST match train/train_nnue.py exactly, or the engine
 * plays against itself.  See also APPENDIX D of the v400 plan.
 *
 *   Zchezz mailbox square  : zsq, 0 = a8 ... 63 = h1
 *   White-POV square       : sq_w = zsq ^ 56   (== python-chess sq, a1=0)
 *   Black-POV square       : sq_b = zsq        (== sq_w ^ 56, vertical mirror)
 *
 *   king_bucket(s) for a POV square s:
 *       bit 0 = (s % 8) >= 4      -> kingside half
 *       bit 1 = (s / 8) >= 4      -> far half (ranks 5-8 in own POV)
 *
 *       bucket_w = king_bucket(wk ^ 56)     wk = white king mailbox sq
 *       bucket_b = king_bucket(bk)          bk = black king mailbox sq
 *
 *   Relative piece index (the KING IS NEVER A FEATURE):
 *       ally   P,N,B,R,Q -> 0,1,2,3,4
 *       enemy  P,N,B,R,Q -> 5,6,7,8,9
 *
 *   feature = bucket * 640 + rel * 64 + pov_square       (0 .. 2559)
 *
 * Both perspectives share the SAME L1 weight matrix.  The concat order
 * is ALWAYS [stm, opp] — swapping it makes the engine evaluate won
 * positions as lost.
 *
 * ── Bucket refresh (lazy, Stockfish-style) ───────────────────────
 *
 * A king move that changes its own bucket invalidates EVERY feature of
 * that perspective (all of them carry the bucket in their index), so
 * the perspective must be rebuilt from the board.  nnue_push_na cannot
 * do it: board_make calls it with the PRE-move board.  So the push only
 * raises a per-frame dirty flag, and nnue_eval* rebuilds on demand from
 * the current (post-move) board.
 *
 * The flag lives in a per-frame stack alongside the bucket, so that:
 *   - it survives further pushes made before any eval (a child frame
 *     inherits the parent's dirty flag: its accumulator is built on top
 *     of a stale one and is equally invalid),
 *   - board_unmake / nnue_pop_na restores both bucket and flag exactly.
 * If a node never calls eval (e.g. a tablebase cutoff), the flag is
 * simply discarded on pop — harmless.
 *
 * ── Thread safety model (unchanged from v3.14) ───────────────────
 *
 * All mutable state lives in NnueAccum, one per search thread:
 *   Main thread  -> static global `g_nnue_accum` (defined in board.c)
 *   SMP helpers  -> heap-allocated via zmalloc32 in main.c
 *   WASM         -> static global (single-threaded)
 * Weight matrices are global, loaded once, never written again — safe
 * to share across threads.
 *
 * NnueAccum grows to ~257 KB (was ~68 KB): the two 128-deep stacks are
 * now 512 wide instead of 256.  Helpers heap-allocate it, so this costs
 * heap, not stack.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

/* ── Per-instance weight set (v4.00 Phase 0-follow-up) ────────────
 *
 * Packaged into an opaque struct, exactly analogous to TTable in
 * search.h: this is what makes it possible to load TWO different NNUE
 * networks into ONE process (native A/B arena: gen_{i+1} vs gen_i).
 * Before this, _nnL1WT/_nnL1B/... were file-scope statics in nnue.c —
 * only one net could ever be loaded per process.
 *
 * The definition lives in nnue.c; callers only ever see a pointer, so
 * they cannot poke at the weight arrays directly (mirrors TTable's
 * "SoA fields are private to search.c's tt_* functions" convention,
 * just enforced by the compiler instead of by comment). */
typedef struct NnueNet NnueNet;

/* Allocate an empty (unloaded) NnueNet. Returns NULL on OOM. */
NnueNet *nnue_net_create(void);

/* create() + load weights from `path` into it. Returns NULL on any
 * failure (OOM or bad file); the partially-built net, if any, is freed
 * before returning so callers never get a half-loaded object. */
NnueNet *nnue_net_load(const char *path);

/* create() + load weights from an in-memory NNU4 buffer. Same NULL
 * contract as nnue_net_load(). */
NnueNet *nnue_net_load_from_mem(const uint8_t *data, size_t len);

/* Free a NnueNet and all of its backing arrays. Safe to call with NULL.
 * Caller is responsible for making sure no live NnueAccum still points
 * at it (na->net becomes a dangling pointer otherwise — exactly like
 * freeing a TTable out from under a SearchParams). */
void nnue_net_destroy(NnueNet *net);

/* True once `net` holds successfully loaded weights. NULL-safe. */
int  nnue_net_ready(const NnueNet *net);

/* Process-wide default net used by the UCI engine / WASM build and by
 * every legacy (non-net-aware) API below. Allocated + loaded by the
 * first successful nnue_load()/nnue_load_from_mem() call; every legacy
 * caller implicitly uses this net, exactly like g_tt in search.h. */
extern NnueNet *g_nnue_net;

/* ── Layer dimensions ──────────────────────────────────────────── */
#define NN_KING_BUCKETS      4
#define NN_FEAT_PER_BUCKET 640   /* 10 relative piece planes x 64 squares */
#define NN_FEAT_IN        2560   /* NN_KING_BUCKETS * NN_FEAT_PER_BUCKET  */
#define NN_L1_IN          2560   /* alias of NN_FEAT_IN (per perspective) */
#define NN_L1_OUT          512
#define NN_L2_IN          1024   /* concat of both perspectives: 2*L1_OUT */
#define NN_L2_OUT           32
#define NN_L3_IN            32   /* must equal NN_L2_OUT                  */
#define NN_L3_OUT            1

/* ── Quantization constants ────────────────────────────────────── */
#define NN_QA      255   /* L1 weight/activation scale                    */
#define NN_QB       64   /* L2/L3 weight scale, L2 activation ceiling     */
#define NN_SHIFT     8   /* L2 accumulator right-shift before ClippedReLU */
/* Effective scale of the SCReLU output: (255*255)>>8 = 254 */
#define NN_QA_EFF  254

/* ── Piece-type index ──────────────────────────────────────────── */
/* P=0 N=1 B=2 R=3 Q=4 (K is deliberately absent — not a feature) */
#define NN_NOTYPES  6
#define NN_NOCOLORS 2

/* ── Accumulator stack depth ───────────────────────────────────── */
/* Must be > MAX_PLY (64); 128 gives comfortable headroom. */
#define NN_ACC_STACK 128

/* Legacy alias kept for any external reference. */
#define NN_ACC_DEPTH NN_ACC_STACK

/* ── Per-thread NNUE accumulator state ─────────────────────────
 *
 * Memory layout (~257 KB, 32-byte aligned for AVX2):
 *   acc_stack_w[128][512]  White-POV accumulator stack   (128 KB)
 *   acc_stack_b[128][512]  Black-POV accumulator stack   (128 KB)
 *   acc_ptr                current frame index
 *   acc_w/acc_b[512]       scratch used by nnue_rebuild  (2 KB)
 *   bucket_w/b_stack[128]  king bucket per frame         (256 B)
 *   dirty_w/b_stack[128]   "perspective needs rebuild"   (256 B)
 *   acc_dirty              whole-accumulator rebuild flag
 *
 * Everything that used to serve the 31 extra features (ext_buf,
 * ext_feat, ext_dirty, cache_key, cache_buf) is gone.
 *
 * `net` (v4.00 Phase 0-follow-up): the weight set this accumulator was
 * built against. An accumulator is meaningless without the net that
 * produced its numbers — binding them together here is what makes it
 * impossible to accidentally evaluate one net's accumulator with a
 * different net's weights (e.g. two arena players sharing a Board by
 * mistake). NULL means "not bound yet"; the forward pass treats that
 * exactly like acc_dirty and returns 0 instead of dereferencing it. */
typedef struct {
    int16_t  acc_stack_w[NN_ACC_STACK][NN_L1_OUT] __attribute__((aligned(32)));
    int16_t  acc_stack_b[NN_ACC_STACK][NN_L1_OUT] __attribute__((aligned(32)));
    int      acc_ptr;
    int16_t  acc_w[NN_L1_OUT]                     __attribute__((aligned(32)));
    int16_t  acc_b[NN_L1_OUT]                     __attribute__((aligned(32)));
    uint8_t  bucket_w_stack[NN_ACC_STACK];
    uint8_t  bucket_b_stack[NN_ACC_STACK];
    uint8_t  dirty_w_stack[NN_ACC_STACK];
    uint8_t  dirty_b_stack[NN_ACC_STACK];
    int      acc_dirty;
    const NnueNet *net;
} NnueAccum;

/* ── Public API ────────────────────────────────────────────────── */

/* Load NNU4 binary from path.  Returns 0 on success, -1 on error.
 * Thread safety: call once from main thread before any search. */
int  nnue_load(const char *path);

/* True after a successful nnue_load(). Thread-safe (read-only flag). */
int  nnue_ready(void);

/* Fully rebuild both perspectives from board[64].
 * Resets acc_ptr to 0 and seeds frame 0's bucket/dirty state.
 * Called at the start of each search and whenever the accumulator
 * stack overflows.
 *
 * na:        per-thread accumulator (Board::nnue)
 * board[64]: piece constants (WP=9..BK=22, empty=0) */
void nnue_rebuild(NnueAccum *na, const uint8_t *board);

/* NNMove — describes a move for incremental accumulator push.
 * wk_sq/bk_sq are the king squares BEFORE the move; board_make fills
 * them from b->wk / b->bk so nnue_push_na never has to scan the board. */
typedef struct {
    uint8_t from_sq;
    uint8_t to_sq;
    uint8_t prom;       /* 0 = not a promotion */
    uint8_t is_epc;     /* en-passant capture flag */
    uint8_t castle;     /* 0|1|2|3|4 */
    uint8_t wk_sq;      /* white king mailbox square before the move */
    uint8_t bk_sq;      /* black king mailbox square before the move */
} NNMove;

/* Per-thread incremental accumulator update — push/pop on na->acc_stack.
 * Called by board_make/board_unmake in the native search path.
 * `board` is the position BEFORE the move.
 * Thread-safe: each thread has its own NnueAccum. */
void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m);
void nnue_pop_na(NnueAccum *na);

/* Legacy global push/pop (WASM / test functions only).
 * Operate on g_nnue_accum. */
void nnue_push(const uint8_t *board, const NNMove *m);
void nnue_pop(void);

/* Mark `na` as needing a full rebuild.  Called at the start of every
 * search (search_best) before nnue_rebuild. */
void nnue_reset(NnueAccum *na);

/* Forward pass.
 *
 * na:        per-thread accumulator
 * stm:       0 = White to move, 1 = Black to move
 * board[64]: current position (needed only for a lazy bucket refresh)
 * Returns:   centipawns, STM-relative, clamped [-2000, +2000] */
int  nnue_eval(NnueAccum *na, int stm, const uint8_t *board);

/* Same forward pass, kept for source compatibility with the search.
 * v4.00 no longer needs bb[] or the Zobrist hash (the extra-feature
 * cache that used them is gone); both arguments are ignored. */
int  nnue_eval_bb(NnueAccum *na, int stm, const uint8_t *board,
                  const uint64_t bb[12], uint64_t board_hash);

/* Load weights from a memory buffer (WebAssembly / browser).
 * data must point to the raw NNU4 binary, len is the byte count.
 * Returns 0 on success, -1 on error.  Thread safety: call once. */
int nnue_load_from_mem(const uint8_t *data, size_t len);

/* WASM backward-compatibility wrapper: JS calls nnue_reset() with no
 * args — this passes the global accumulator. */
void nnue_reset_global(void);

/* ── Feature encoding, exposed for tests / tools ──────────────────
 * nnue_feature_index: HalfKP-4Bucket index (0..2559) of piece `p` on
 * mailbox square `zsq` from the given perspective, or -1 if `p` is
 * empty or a king.  white_pov: 1 = White perspective, 0 = Black.
 * `bucket` must be that perspective's king bucket. */
int nnue_feature_index(uint8_t p, int zsq, int white_pov, int bucket);

/* King bucket of each perspective, from mailbox king squares. */
int nnue_king_bucket_w(int wk_zsq);
int nnue_king_bucket_b(int bk_zsq);
