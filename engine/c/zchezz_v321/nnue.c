#define _POSIX_C_SOURCE 200809L
/* nnue.c — Zchezz NNUE evaluation layer (v3.14: per-thread acc stacks)
 *
 * v3.14 ARCHITECTURE: HM accumulator (768 half-king-piece features) uses
 * per-thread acc_stack_w/b in NnueAccum.  board_make/unmake call
 * nnue_push_na/pop_na.  nnue_eval_bb reads from na->acc_stack_w[acc_ptr].
 * Legacy nnue_push/pop still update global _acc_buf for WASM/tests.
 * Weight tables remain global (read-only after load, safe for MT).
 *
 * KEY FIXES vs previous version:
 *
 * FIX 1 (MAJOR SPEEDUP – L2 kernel rewritten with maddubs_epi16):
 *   Old code: cvtepi8_epi32 + mullo_epi32 = processes 8 int8 weights per
 *   256-bit register, one output column at a time.
 *   New code: _mm256_maddubs_epi16 + _mm256_madd_epi16 = processes 32
 *   uint8 × int8 products per register. For 256-input × 64-output L2 this
 *   is ~4× more SIMD work per instruction. This is the canonical NNUE kernel
 *   used by Stockfish and Leela.
 *   L2 layout changed: [out][in] (row-major per output neuron) so we can
 *   iterate outputs in the outer loop and accumulate all 256 inputs at once
 *   using AVX2. This also makes the weight rows contiguous for prefetching.
 *
 * FIX 2 (CORRECTNESS + SPEED – aligned weight allocation):
 *   All weight arrays now allocated with 32-byte alignment (posix_memalign /
 *   aligned_alloc). This allows _mm256_load_si256 (vs loadu) in all loops —
 *   unaligned loads cost 1-3 extra cycles each on Intel, and misaligned
 *   _mm256_load_si256 crashes. Stack buffers already had __attribute__((aligned(32))).
 *
 * FIX 3 (MEDIUM SPEEDUP – L3 dot product is integer, not float):
 *   Old: _nnL3W[i] * (float)relu2[i] accumulated in float.
 *   New: integer accumulation → single int32 → one float convert at the end.
 *   Eliminates 64 int→float conversions from the hot path.
 *
 * FIX 4 (CORRECTNESS – nnue_push/pop wired into search):
 *   nnue_push and nnue_pop are not called by search.c (board_make/unmake
 *   don't call them). The accumulator at _acc_buf[0] is therefore always the
 *   ROOT position. nnue_eval reads _acc_buf[_acc_ptr] where _acc_ptr stays 0.
 *   This is fine IF the caller rebuilds before each root. The _ext_buf
 *   caching is still useful: with _acc_ptr==0 always, _ext_dirty[0][stm]
 *   starts true (via nnue_reset), gets computed once on the first eval of a
 *   given stm, then cached for the rest of that depth iteration.
 *   No change needed here — but this explains why incremental push/pop
 *   doesn't help speed if search never calls them. Adding push/pop calls to
 *   board_make/unmake in search.c is a separate, larger win (not in this file).
 *
 * FIX 5 (MEDIUM SPEEDUP – L2 weight layout changed):
 *   Python converter must emit L2W as [out][in] = [64][256] NOT [in][out].
 *   See convert_nnue3.py change: remove the transpose of L2W before writing.
 *   The file format version stays NNU3 but the semantics of L2W_T changed:
 *   it is now row-major-per-output (standard gemv order).
 *   Converter diff:
 *     OLD: L2W_T = np.ascontiguousarray(L2W_q.T)  # [256, 64] int8
 *     NEW: L2W_T = np.ascontiguousarray(L2W_q)    # [64, 256] int8  (no transpose!)
 *
 * Compile:
 *   gcc -O3 -march=native -std=c11 -o test_nnue nnue.c -DNNUE_TEST -lm
 */

#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#  include <immintrin.h>
#elif defined(__wasm_simd128__)
#  include <wasm_simd128.h>
#endif

/* ── Piece constant helpers ──────────────────────────────────────── */
#define PC_COLOR(p)  ((p) & 24)
#define PC_TYPE(p)   ((p) &  7)
#define COL_W  8
#define COL_B 16

static inline int piece_type_idx(uint8_t p) {
    if (p < 9 || p > 22) return -1;
    int t = PC_TYPE(p);
    if (t < 1 || t > 6) return -1;
    return t - 1;
}

/* ── Aligned malloc helper ───────────────────────────────────────── */
static void *zmalloc32(size_t bytes) {
    void *ptr = NULL;
#if defined(_WIN32)
    ptr = _aligned_malloc(bytes, 32);
#elif defined(__APPLE__) || defined(__linux__)
    if (posix_memalign(&ptr, 32, bytes) != 0) ptr = NULL;
#else
    ptr = malloc(bytes);   /* fallback — may cause unaligned AVX2 faults */
#endif
    return ptr;
}

static void zfree32(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ── Weight storage ──────────────────────────────────────────────── */
/* L1: [NN_L1_IN][NN_L1_OUT] = [799][256] int16, 32-byte aligned */
static int16_t *_nnL1WT   = NULL;
static int32_t *_nnL1B    = NULL;  /* [256] int32 */

/* L2: [NN_L2_OUT][NN_L2_IN] = [64][256] int8
 * Row-major per OUTPUT neuron — optimal for the maddubs kernel below.
 * NOTE: This is NOT transposed vs the Python weight matrix [64,256].
 *       The old code transposed to [256,64]; this version does NOT transpose.
 *       See FIX 5 in the header comment. */
static int8_t  *_nnL2W    = NULL;  /* [NN_L2_OUT][NN_L2_IN] = [64][256] */
static int32_t *_nnL2B    = NULL;  /* [64] int32 */

/* L3 */
static int8_t  *_nnL3W    = NULL;  /* [64] int8 */
static float    _nnL3B    = 0.0f;
static float    _nnOutScale = 1.0f;

static int _nnue_ready = 0;
/* External symbol required by nnue.h — kept for linker */
int nnue_ready(void) { return _nnue_ready; }
/* Internal callers use the macro for zero call overhead */
#define nnue_ready() (_nnue_ready)

/* Forward-declared here so _load_weights can reset them */
int _acc_dirty = 1;
int _acc_ptr   = 0;

/* ── NNU3 binary loader ──────────────────────────────────────────── */
int nnue_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[NNUE] Cannot open: %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(fsize);
    if (!buf) { fclose(f); return -1; }
    if ((long)fread(buf, 1, fsize, f) != fsize) { free(buf); fclose(f); return -1; }
    fclose(f);
    int r = nnue_load_from_mem(buf, (size_t)fsize);
    free(buf);
    if (r == 0)
        fprintf(stderr, "[NNUE] Loaded: %s (NNU3 int16/int8) OutScale=%g\n", path, _nnOutScale);
    return r;
}

static int _load_weights(const uint8_t *buf, size_t len) {
    if (len < 8 || buf[0]!='N'||buf[1]!='N'||buf[2]!='U'||buf[3]!='3') {
        fprintf(stderr, "[NNUE] Bad magic (expected NNU3)\n"); return -1;
    }
    size_t off = 4 + 4 + 5*4;  /* magic + epoch + 5 dims */
    float scales[4];
    memcpy(scales, buf+off, 4*sizeof(float)); off += 16;
    _nnOutScale = scales[3];

    const int L1_SZ = NN_L1_IN  * NN_L1_OUT;  /* 799*256 int16 */
    const int L2_SZ = NN_L2_OUT * NN_L2_IN;   /* 64*256  int8  */
    long need = (long)off + L1_SZ*2 + NN_L1_OUT*4 + L2_SZ*1 + NN_L2_OUT*4 + NN_L3_IN*1 + 4;
    if ((long)len < need) { fprintf(stderr, "[NNUE] File too small (%zu < %ld)\n", len, need); return -1; }

    zfree32(_nnL1WT); zfree32(_nnL1B);
    zfree32(_nnL2W);  zfree32(_nnL2B);
    zfree32(_nnL3W);

    _nnL1WT = (int16_t *)zmalloc32(L1_SZ       * sizeof(int16_t));
    _nnL1B  = (int32_t *)zmalloc32(NN_L1_OUT   * sizeof(int32_t));
    _nnL2W  = (int8_t  *)zmalloc32(L2_SZ       * sizeof(int8_t));
    _nnL2B  = (int32_t *)zmalloc32(NN_L2_OUT   * sizeof(int32_t));
    _nnL3W  = (int8_t  *)zmalloc32(NN_L3_IN    * sizeof(int8_t));

    if (!_nnL1WT||!_nnL1B||!_nnL2W||!_nnL2B||!_nnL3W) {
        fprintf(stderr, "[NNUE] malloc failed\n"); return -1;
    }

    memcpy(_nnL1WT, buf+off, L1_SZ     *2); off += L1_SZ*2;
    memcpy(_nnL1B,  buf+off, NN_L1_OUT *4); off += NN_L1_OUT*4;

    /* L2W: file stores [NN_L2_IN][NN_L2_OUT] = [256][64] int8 (old "transposed" format).
     * We need [NN_L2_OUT][NN_L2_IN] = [64][256] for the row-major maddubs kernel.
     * Transpose on load — one-time cost, negligible vs search time. */
    {
        const int8_t *src = (const int8_t *)(buf + off);
        for (int o = 0; o < NN_L2_OUT; o++)
            for (int i = 0; i < NN_L2_IN; i++)
                _nnL2W[o * NN_L2_IN + i] = src[i * NN_L2_OUT + o];
    }
    off += L2_SZ;

    memcpy(_nnL2B,  buf+off, NN_L2_OUT *4); off += NN_L2_OUT*4;
    memcpy(_nnL3W,  buf+off, NN_L3_IN  *1); off += NN_L3_IN;
    memcpy(&_nnL3B, buf+off, 4);

    _nnue_ready = 1;
    _acc_dirty  = 1;
    _acc_ptr    = 0;
    return 0;
}

int nnue_load_from_mem(const uint8_t *data, size_t len) {
    return _load_weights(data, len);
}

/* g_nnue_accum is defined in board.c (extern declared in board.h).
 * Main thread uses it via board_bind_nnue_global(&g_board).
 * SMP helpers allocate their own with zmalloc32 in main.c.
 * We can't include board.h here (circular dep), so forward-declare. */
extern NnueAccum g_nnue_accum;

/* ── Legacy accumulator stacks (push/pop API only) ──────────────── */
/* These remain global because nnue_push/nnue_pop are legacy test
 * functions NOT used by the search.  Search uses NnueAccum.
 * If push/pop is ever wired into search, these move to NnueAccum. */
static int16_t _acc_buf_w[NN_ACC_DEPTH][NN_L1_OUT] __attribute__((aligned(32)));
static int16_t _acc_buf_b[NN_ACC_DEPTH][NN_L1_OUT] __attribute__((aligned(32)));
static int16_t _ext_buf  [NN_ACC_DEPTH][2][NN_L1_OUT] __attribute__((aligned(32)));
static int8_t  _ext_dirty_legacy[NN_ACC_DEPTH][2];
static float   _ext_feat_legacy[NN_ACC_DEPTH][2][NN_EXTRA];

/* ── Reset per-thread NNUE accumulator (v3.13) ──────────────────── */
/* Called at the start of every search (search_best) to mark the
 * accumulator dirty (needs rebuild) and clear the ext cache.
 * Thread safe: operates only on the caller's NnueAccum.
 *
 * Also resets legacy globals so that standalone nnue_eval() after
 * nnue_reset() without nnue_rebuild() returns 0 safely. */
void nnue_reset(NnueAccum *na) {
    /* Per-thread state */
    na->acc_dirty = 1;
    na->acc_ptr   = 0;
    na->ext_dirty[0] = 1;
    na->ext_dirty[1] = 1;
    memset(na->cache_key, 0, sizeof(na->cache_key));
    /* Legacy globals (for WASM/test push/pop API) */
    _acc_ptr   = 0;
    _acc_dirty = 1;
    memset(_ext_dirty_legacy, 1, sizeof(_ext_dirty_legacy));
}

/* WASM backward-compatibility: JS worker calls nnue_reset() with no args.
 * This wrapper passes the global accumulator.  Exported as _nnue_reset
 * in the WASM build. */
void nnue_reset_global(void) {
    nnue_reset(&g_nnue_accum);
}

/* ── Feature index helpers (accumulator add/sub) ─────────────────── */
static inline void _acc_add_piece(int16_t *accW, int16_t *accB, uint8_t p, int sq) {
    int pt = piece_type_idx(p); if (pt < 0) return;
    int isW  = (PC_COLOR(p) == COL_W);
    int pySq = sq ^ 56;
    int coW  = isW ? 0 : 6, coB = isW ? 6 : 0;
    const int16_t *wRow = _nnL1WT + (coW*64 + pt*64 + pySq) * NN_L1_OUT;
    const int16_t *bRow = _nnL1WT + (coB*64 + pt*64 + sq  ) * NN_L1_OUT;
#ifdef __AVX2__
    for (int o = 0; o < NN_L1_OUT; o += 16) {
        __m256i aw = _mm256_load_si256 ((const __m256i*)(accW + o));
        __m256i ab = _mm256_load_si256 ((const __m256i*)(accB + o));
        aw = _mm256_add_epi16(aw, _mm256_load_si256((const __m256i*)(wRow + o)));
        ab = _mm256_add_epi16(ab, _mm256_load_si256((const __m256i*)(bRow + o)));
        _mm256_store_si256((__m256i*)(accW + o), aw);
        _mm256_store_si256((__m256i*)(accB + o), ab);
    }
#elif defined(__wasm_simd128__)
    /* WASM SIMD128: 128-bit registers hold 8× int16, so stride=8 (vs 16 for AVX2).
     * Same logic: load acc, add row, store. NN_L1_OUT=256, so 32 iterations. */
    for (int o = 0; o < NN_L1_OUT; o += 8) {
        v128_t aw = wasm_v128_load(accW + o);
        v128_t ab = wasm_v128_load(accB + o);
        aw = wasm_i16x8_add(aw, wasm_v128_load(wRow + o));
        ab = wasm_i16x8_add(ab, wasm_v128_load(bRow + o));
        wasm_v128_store(accW + o, aw);
        wasm_v128_store(accB + o, ab);
    }
#else
    for (int o = 0; o < NN_L1_OUT; o++) { accW[o] += wRow[o]; accB[o] += bRow[o]; }
#endif
}

static inline void _acc_sub_piece(int16_t *accW, int16_t *accB, uint8_t p, int sq) {
    int pt = piece_type_idx(p); if (pt < 0) return;
    int isW  = (PC_COLOR(p) == COL_W);
    int pySq = sq ^ 56;
    int coW  = isW ? 0 : 6, coB = isW ? 6 : 0;
    const int16_t *wRow = _nnL1WT + (coW*64 + pt*64 + pySq) * NN_L1_OUT;
    const int16_t *bRow = _nnL1WT + (coB*64 + pt*64 + sq  ) * NN_L1_OUT;
#ifdef __AVX2__
    for (int o = 0; o < NN_L1_OUT; o += 16) {
        __m256i aw = _mm256_load_si256 ((const __m256i*)(accW + o));
        __m256i ab = _mm256_load_si256 ((const __m256i*)(accB + o));
        aw = _mm256_sub_epi16(aw, _mm256_load_si256((const __m256i*)(wRow + o)));
        ab = _mm256_sub_epi16(ab, _mm256_load_si256((const __m256i*)(bRow + o)));
        _mm256_store_si256((__m256i*)(accW + o), aw);
        _mm256_store_si256((__m256i*)(accB + o), ab);
    }
#elif defined(__wasm_simd128__)
    for (int o = 0; o < NN_L1_OUT; o += 8) {
        v128_t aw = wasm_v128_load(accW + o);
        v128_t ab = wasm_v128_load(accB + o);
        aw = wasm_i16x8_sub(aw, wasm_v128_load(wRow + o));
        ab = wasm_i16x8_sub(ab, wasm_v128_load(bRow + o));
        wasm_v128_store(accW + o, aw);
        wasm_v128_store(accB + o, ab);
    }
#else
    for (int o = 0; o < NN_L1_OUT; o++) { accW[o] -= wRow[o]; accB[o] -= bRow[o]; }
#endif
}

/* ── Extra-feature (31 endgame features) computation ────────────── */

/* Precomputed bitboard masks for passed-pawn detection.
 *
 * _pp_span_w[sq] : all squares strictly ABOVE sq (smaller row index) on the
 *                  same file and adjacent files — i.e. the "front span" for a
 *                  White pawn on sq.  A White pawn on sq is passed if no Black
 *                  pawn sits in this mask.
 * _pp_span_b[sq] : mirror for Black (squares strictly BELOW sq).
 *
 * Square layout: 0=a8 (top-left), 63=h1 (bottom-right).
 * Row increases downward: sq=r*8+f, r=0 is rank 8, r=7 is rank 1.
 *
 * White pawns advance toward smaller r (rank 8 direction).
 * Black pawns advance toward larger r (rank 1 direction).
 */
static uint64_t _pp_span_w[64];  /* front span for White pawn (rows < r, files f-1..f+1) */
static uint64_t _pp_span_b[64];  /* front span for Black pawn (rows > r, files f-1..f+1) */
static int      _extra_masks_init = 0;

/* Build file-trio mask: all squares on files (f-1, f, f+1) */
static uint64_t _file_trio(int f) {
    uint64_t mask = 0;
    for (int r=0; r<8; r++) {
        if (f > 0) mask |= (uint64_t)1 << (r*8 + f-1);
        mask |= (uint64_t)1 << (r*8 + f);
        if (f < 7) mask |= (uint64_t)1 << (r*8 + f+1);
    }
    return mask;
}

static void _init_extra_masks(void) {
    if (_extra_masks_init) return;
    for (int sq=0; sq<64; sq++) {
        int r = sq >> 3, f = sq & 7;
        uint64_t trio = _file_trio(f);
        /* White: rows 0..r-1 (above sq) */
        uint64_t above = 0;
        for (int rr=0; rr<r; rr++) {
            if (f>0) above |= (uint64_t)1 << (rr*8 + f-1);
            above |= (uint64_t)1 << (rr*8 + f);
            if (f<7) above |= (uint64_t)1 << (rr*8 + f+1);
        }
        _pp_span_w[sq] = above;
        /* Black: rows r+1..7 (below sq) */
        uint64_t below = 0;
        for (int rr=r+1; rr<8; rr++) {
            if (f>0) below |= (uint64_t)1 << (rr*8 + f-1);
            below |= (uint64_t)1 << (rr*8 + f);
            if (f<7) below |= (uint64_t)1 << (rr*8 + f+1);
        }
        _pp_span_b[sq] = below;
        (void)trio;
    }
    _extra_masks_init = 1;
}

/* Build bitboard representation from board[64] array.
 * bb[0..5]  = White P N B R Q K
 * bb[6..11] = Black P N B R Q K
 *
 * Called only inside _compute_extra_feat — inlined by the compiler.
 * Cost: 64-iteration loop with 1 branch + 2 shifts per occupied square.
 * This replaces two separate 64-square loops in the old code. */
static inline void _build_bb_from_board(const uint8_t *board, uint64_t bb[12]) {
    memset(bb, 0, 12*sizeof(uint64_t));
    for (int sq=0; sq<64; sq++) {
        uint8_t p = board[sq]; if (!p) continue;
        int t = PC_TYPE(p)-1; if ((unsigned)t > 5u) continue;
        int idx = (PC_COLOR(p)==COL_W) ? t : t+6;
        bb[idx] |= (uint64_t)1 << sq;
    }
}

/* Compute the raw 31 feature values into feat[] from scratch.
 *
 * BITBOARD VERSION — replaces the original O(64²) scalar loops.
 *
 * Piece counts   : popcount on 12 bitboards — O(12) vs O(64)
 * King squares   : ctzll on king BB        — O(1)  vs O(64)
 * Passed pawns   : span-mask AND test      — O(pawns) with no inner loop
 *                  (old: up to 16 pawns × 3 files × 7 ranks = 336 branches)
 *
 * The board[] array is still the authoritative source (NNUE eval path does
 * not have a Board* with precomputed bb[]).  We build the 12 BB in one pass
 * then do all feature extraction with integer operations. */
static void _compute_extra_feat(float *feat, const uint8_t *board, int stm) {
    static const float MAXCNT[6] = {8.f,2.f,2.f,2.f,1.f,1.f};
    static const float MATVAL[6] = {1.f,3.f,3.f,5.f,9.f,0.f};

    if (!_extra_masks_init) _init_extra_masks();

    /* Build bitboards in one O(64) pass */
    uint64_t bb[12];
    _build_bb_from_board(board, bb);

    /* ── Piece counts via popcount ──────────────────────────────── */
    int cnt_w[6], cnt_b[6];
    for (int t=0; t<6; t++) {
        cnt_w[t] = __builtin_popcountll(bb[t]);
        cnt_b[t] = __builtin_popcountll(bb[t+6]);
    }

    int *stm_cnt = (stm==0) ? cnt_w : cnt_b;
    int *opp_cnt = (stm==0) ? cnt_b : cnt_w;
    for (int i=0;i<6;i++) feat[i]   = stm_cnt[i]/MAXCNT[i];
    for (int i=0;i<6;i++) feat[6+i] = opp_cnt[i]/MAXCNT[i];

    float mat=0.f;
    for (int i=0;i<6;i++) mat += (cnt_w[i]+cnt_b[i])*MATVAL[i];
    feat[12] = mat/78.f;
    feat[13] = 1.0f;

    /* ── King squares via ctzll ─────────────────────────────────── */
    int wk_sq = bb[5]  ? __builtin_ctzll(bb[5])  : -1;
    int bk_sq = bb[11] ? __builtin_ctzll(bb[11]) : -1;

    /* ── Passed pawns via bitboard span masks ───────────────────── */
    /* feat[14..21]: stm passed-pawn flags per file (any pawn = 1.0)
     * feat[22..29]: opp passed-pawn flags per file
     *
     * A White pawn on sq is passed if (_pp_span_w[sq] & bb_black_pawns) == 0.
     * A Black pawn on sq is passed if (_pp_span_b[sq] & bb_white_pawns) == 0.
     * stm/opp mapping flips when stm==1. */
    for (int f=0; f<8; f++) { feat[14+f]=0.f; feat[22+f]=0.f; }

    uint64_t wp = bb[0], bp = bb[6];

    if (stm == 0) {
        /* stm=White: feat[14+file]=1 if White has passed pawn on that file */
        uint64_t tmp = wp;
        while (tmp) {
            int sq = __builtin_ctzll(tmp); tmp &= tmp-1;
            if (!(_pp_span_w[sq] & bp)) feat[14 + (sq&7)] = 1.f;
        }
        /* opp=Black: feat[22+file]=1 if Black has passed pawn on that file */
        tmp = bp;
        while (tmp) {
            int sq = __builtin_ctzll(tmp); tmp &= tmp-1;
            if (!(_pp_span_b[sq] & wp)) feat[22 + (sq&7)] = 1.f;
        }
    } else {
        /* stm=Black: feat[14+file]=1 if Black has passed pawn on that file */
        uint64_t tmp = bp;
        while (tmp) {
            int sq = __builtin_ctzll(tmp); tmp &= tmp-1;
            if (!(_pp_span_b[sq] & wp)) feat[14 + (sq&7)] = 1.f;
        }
        /* opp=White: feat[22+file]=1 if White has passed pawn on that file */
        tmp = wp;
        while (tmp) {
            int sq = __builtin_ctzll(tmp); tmp &= tmp-1;
            if (!(_pp_span_w[sq] & bp)) feat[22 + (sq&7)] = 1.f;
        }
    }

    /* ── Chebyshev king distance ─────────────────────────────────── */
    if (wk_sq>=0 && bk_sq>=0) {
        int wf=wk_sq&7, wr=wk_sq>>3, bf=bk_sq&7, br=bk_sq>>3;
        int df=wf>bf?wf-bf:bf-wf, dr=wr>br?wr-br:br-wr;
        feat[30]=(float)(df>dr?df:dr)/7.f;
    } else feat[30]=0.f;
}

/* Project a single feature column j with value fj into out[].
 * out[o] += round(fj * L1WT[NN_HM_IN+j][o])  (fixed-point, shift-8)
 *
 * Fast path: fj_16==256 means feat==1.0 (binary feature, very common for
 * pawn-file flags, piece-present counts, and the always-1 bias feature).
 * In that case the multiply/shift collapses to a plain add — saves 1 instruction
 * per 16-wide AVX2 lane, ~30% faster for the dominant binary features. */
static inline void _project_feat_add(int16_t *out, int j, int16_t fj_16) {
    const int16_t *row = _nnL1WT + (NN_HM_IN+j)*NN_L1_OUT;
#ifdef __AVX2__
    if (fj_16 == 256) {
        /* Binary feature: just add row directly, no multiply */
        for (int o=0; o<NN_L1_OUT; o+=16) {
            __m256i r = _mm256_load_si256((const __m256i*)(row + o));
            __m256i v = _mm256_load_si256((const __m256i*)(out + o));
            _mm256_store_si256((__m256i*)(out + o), _mm256_add_epi16(v, r));
        }
    } else {
        __m256i v_fj = _mm256_set1_epi16(fj_16);
        for (int o=0; o<NN_L1_OUT; o+=16) {
            __m256i row_16 = _mm256_load_si256((const __m256i*)(row + o));
            __m256i prod   = _mm256_mullo_epi16(v_fj, row_16);
            prod = _mm256_srai_epi16(prod, 8);
            __m256i out_16 = _mm256_load_si256((const __m256i*)(out + o));
            _mm256_store_si256((__m256i*)(out + o), _mm256_add_epi16(out_16, prod));
        }
    }
#elif defined(__wasm_simd128__)
    /* WASM SIMD128: v128 holds 8× int16, stride=8.
     * Binary fast path: fj_16==256 means feat==1.0, skip multiply. */
    if (fj_16 == 256) {
        for (int o=0; o<NN_L1_OUT; o+=8) {
            v128_t r = wasm_v128_load(row + o);
            v128_t v = wasm_v128_load(out + o);
            wasm_v128_store(out + o, wasm_i16x8_add(v, r));
        }
    } else {
        v128_t v_fj = wasm_i16x8_splat(fj_16);
        for (int o=0; o<NN_L1_OUT; o+=8) {
            v128_t row_16 = wasm_v128_load(row + o);
            /* mullo_i16: low 16 bits of each 16×16 product */
            v128_t prod   = wasm_i16x8_mul(v_fj, row_16);
            prod = wasm_i16x8_shr(prod, 8);   /* arithmetic right-shift by 8 */
            v128_t out_16 = wasm_v128_load(out + o);
            wasm_v128_store(out + o, wasm_i16x8_add(out_16, prod));
        }
    }
#else
    int32_t fj_fp = (int32_t)fj_16;
    for (int o=0; o<NN_L1_OUT; o++) out[o] += (int16_t)((fj_fp * row[o]) >> 8);
#endif
}

/* Build ext_buf from scratch given a precomputed feat[] array.
 * Called on nnue_rebuild (full recompute). */
static void _project_feat_full(int16_t *out, const float *feat) {
    memset(out, 0, NN_L1_OUT*sizeof(int16_t));
    for (int j=0; j<NN_EXTRA; j++) {
        if (feat[j]==0.f) continue;
        int16_t fj_16 = (int16_t)(feat[j] * 256.0f);
        _project_feat_add(out, j, fj_16);
    }
}

/* Incremental update: given old_feat[] stored in parent slot and new_feat[]
 * for the child slot, copy parent ext_buf then apply only the delta columns.
 * This avoids the O(31 × 256) SIMD work for unchanged features. */
static void _project_feat_incremental(int16_t *out, const int16_t *parent_out,
                                       const float *old_feat, const float *new_feat) {
    memcpy(out, parent_out, NN_L1_OUT*sizeof(int16_t));
    for (int j=0; j<NN_EXTRA; j++) {
        float delta = new_feat[j] - old_feat[j];
        if (delta == 0.f) continue;
        int16_t delta_16 = (int16_t)(delta * 256.0f);
        if (delta_16 == 0) continue;
        _project_feat_add(out, j, delta_16);
    }
}

/* Legacy full-recompute wrapper (kept for nnue_eval's lazy path when
 * _ext_feat was never seeded — e.g. after nnue_reset without rebuild). */
static void _compute_extra_acc(int16_t *out, const uint8_t *board, int stm) {
    float feat[NN_EXTRA];
    _compute_extra_feat(feat, board, stm);
    _project_feat_full(out, feat);
}

/* ── Rebuild accumulator from scratch (v3.13: per-thread NnueAccum) ── */
/* Scans all 768 HM features from board[64], writes acc_w/acc_b.
 * Called once at the start of each iterative deepening iteration.
 *
 * Thread safe: writes only to the caller's NnueAccum.
 * Also syncs legacy _acc_buf_w/b[0] for push/pop test compatibility. */
void nnue_rebuild(NnueAccum *na, const uint8_t *board) {
    int16_t *dW = na->acc_w, *dB = na->acc_b;
    memset(dW, 0, NN_L1_OUT*sizeof(int16_t));
    memset(dB, 0, NN_L1_OUT*sizeof(int16_t));
    for (int sq=0; sq<64; sq++) {
        uint8_t p=board[sq]; if (!p) continue;
        _acc_add_piece(dW,dB,p,sq);
    }
    na->acc_dirty = 0;
    /* v3.14: copy rebuild result to per-thread acc_stack[0] */
    na->acc_ptr = 0;
    memcpy(na->acc_stack_w[0], dW, NN_L1_OUT*sizeof(int16_t));
    memcpy(na->acc_stack_b[0], dB, NN_L1_OUT*sizeof(int16_t));
    /* Seed extra-feature arrays in NnueAccum for both stm directions */
    for (int stm=0; stm<2; stm++) {
        _compute_extra_feat(na->ext_feat[stm], board, stm);
        memset(na->ext_buf[stm], 0, NN_L1_OUT*sizeof(int16_t));
        _project_feat_full(na->ext_buf[stm], na->ext_feat[stm]);
        na->ext_dirty[stm] = 0;
    }
}

/* ── Castle square tables ────────────────────────────────────────── */
static const int _castle_sq[5][4] = {
    {0,0,0,0},{60,62,63,61},{60,58,56,59},{4,6,7,5},{4,2,0,3},
};

/* ── Push / Pop (per-thread acc_stack) ── */
/* Called by board_make/board_unmake via nnue_push_na/pop_na.
 * Thread-safe: each thread has its own NnueAccum with its own stack. */
void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m) {
    int src = na->acc_ptr, dst = src + 1;
    if (dst >= NN_ACC_STACK) { na->acc_dirty = 1; return; }
    if (na->acc_dirty) { nnue_rebuild(na, board); na->acc_dirty = 0; src = 0; dst = 1; }
    memcpy(na->acc_stack_w[dst], na->acc_stack_w[src], NN_L1_OUT*sizeof(int16_t));
    memcpy(na->acc_stack_b[dst], na->acc_stack_b[src], NN_L1_OUT*sizeof(int16_t));
    int16_t *cW = na->acc_stack_w[dst], *cB = na->acc_stack_b[dst];
    if (m->castle) {
        const int *sq = _castle_sq[m->castle];
        _acc_sub_piece(cW, cB, board[sq[0]], sq[0]);
        _acc_add_piece(cW, cB, board[sq[0]], sq[1]);
        _acc_sub_piece(cW, cB, board[sq[2]], sq[2]);
        _acc_add_piece(cW, cB, board[sq[2]], sq[3]);
    } else {
        int f = m->from_sq, to = m->to_sq;
        uint8_t p = board[f], cap = board[to];
        _acc_sub_piece(cW, cB, p, f);
        if (cap) _acc_sub_piece(cW, cB, cap, to);
        if (m->is_epc) {
            int epsq = (PC_COLOR(p) == COL_W) ? to + 8 : to - 8;
            if (board[epsq]) _acc_sub_piece(cW, cB, board[epsq], epsq);
        }
        uint8_t landing = m->prom ? (uint8_t)(PC_COLOR(p) | m->prom) : p;
        _acc_add_piece(cW, cB, landing, to);
    }
    na->acc_ptr = dst;
    na->ext_dirty[0] = 1;
    na->ext_dirty[1] = 1;
}

void nnue_pop_na(NnueAccum *na) { if (na->acc_ptr > 0) na->acc_ptr--; }

/* ── Legacy Push / Pop (operates on global arrays, WASM/test only) ── */
void nnue_push(const uint8_t *board, const NNMove *m) {
    int src=_acc_ptr, dst=src+1;
    if (dst>=NN_ACC_DEPTH) { _acc_dirty=1; return; }
    if (_acc_dirty) {
        nnue_rebuild(&g_nnue_accum, board);
        /* Sync globals since nnue_rebuild no longer does it */
        _acc_dirty = 0;
        _acc_ptr   = 0;
        memcpy(_acc_buf_w[0], g_nnue_accum.acc_stack_w[0], NN_L1_OUT*sizeof(int16_t));
        memcpy(_acc_buf_b[0], g_nnue_accum.acc_stack_b[0], NN_L1_OUT*sizeof(int16_t));
    }
    memcpy(_acc_buf_w[dst], _acc_buf_w[src], NN_L1_OUT*sizeof(int16_t));
    memcpy(_acc_buf_b[dst], _acc_buf_b[src], NN_L1_OUT*sizeof(int16_t));
    int16_t *cW=_acc_buf_w[dst], *cB=_acc_buf_b[dst];
    if (m->castle) {
        const int *sq=_castle_sq[m->castle];
        _acc_sub_piece(cW,cB,board[sq[0]],sq[0]);
        _acc_add_piece(cW,cB,board[sq[0]],sq[1]);
        _acc_sub_piece(cW,cB,board[sq[2]],sq[2]);
        _acc_add_piece(cW,cB,board[sq[2]],sq[3]);
    } else {
        int f=m->from_sq, to=m->to_sq;
        uint8_t p=board[f], cap=board[to];
        _acc_sub_piece(cW,cB,p,f);
        if (cap) _acc_sub_piece(cW,cB,cap,to);
        if (m->is_epc) {
            int epsq=(PC_COLOR(p)==COL_W)?to+8:to-8;
            if(board[epsq]) _acc_sub_piece(cW,cB,board[epsq],epsq);
        }
        uint8_t landing=m->prom?(uint8_t)(PC_COLOR(p)|m->prom):p;
        _acc_add_piece(cW,cB,landing,to);
    }
    _acc_ptr=dst;
    _ext_dirty_legacy[dst][0]=1;
    _ext_dirty_legacy[dst][1]=1;
}

void nnue_pop(void) { if (_acc_ptr>0) _acc_ptr--; }



/* ════════════════════════════════════════════════════════════════════
 * FORWARD PASS — nnue_eval
 *
 * Step 1  : Lazy-compute extra-feature delta if dirty.
 * Step 2  : relu1[256] = ClippedReLU(acc_HM + ext + bias, 0, 255) as uint8.
 * Step 3  : L2 — 64 neurons, [64][256] int8 weights.
 *           FAST KERNEL: for each output o, compute dot(relu1, L2W[o])
 *           using _mm256_maddubs_epi16 (uint8×int8→int16 pairwise+add)
 *           + _mm256_madd_epi16 (int16×1 →int32 pairwise+add) to get int32.
 *           Process 32 input elements per 256-bit register.
 * Step 4  : shift + ClippedReLU → relu2[64] uint8.
 * Step 5  : L3 — int32 dot product, single float convert.
 * Step 6  : scale + bias → cp.
 * ════════════════════════════════════════════════════════════════════ */

#ifdef __AVX2__
/* Sum all 8 int32 lanes of a __m256i register into one int32 */
static inline int32_t _hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    return _mm_cvtsi128_si32(sum);
}
#endif

int nnue_eval(NnueAccum *na, int stm, const uint8_t *board) {
    if (!_nnue_ready) return 0;
    if (na->acc_dirty) return 0;  /* v3.14: per-thread dirty flag */

    /* Step 1: compute extra feature values and project into ext_buf */
    {
        _compute_extra_feat(na->ext_feat[stm], board, stm);
        memset(na->ext_buf[stm], 0, NN_L1_OUT*sizeof(int16_t));
        _project_feat_full(na->ext_buf[stm], na->ext_feat[stm]);
    }

    /* Step 2: ClippedReLU → relu1[256] uint8 */
    uint8_t relu1[NN_L1_OUT] __attribute__((aligned(32)));
    const int16_t *acc = stm==0 ? na->acc_stack_w[na->acc_ptr] : na->acc_stack_b[na->acc_ptr];
    const int16_t *ext = na->ext_buf[stm];

#ifdef __AVX2__
    {
        __m256i v255 = _mm256_set1_epi32(255);
        __m256i zero = _mm256_setzero_si256();
        for (int o = 0; o < NN_L1_OUT; o += 16) {
            __m128i a_lo = _mm_load_si128((const __m128i*)(acc + o));
            __m128i a_hi = _mm_load_si128((const __m128i*)(acc + o + 8));
            __m128i e_lo = _mm_load_si128((const __m128i*)(ext + o));
            __m128i e_hi = _mm_load_si128((const __m128i*)(ext + o + 8));

            __m256i s_lo = _mm256_add_epi32(_mm256_cvtepi16_epi32(a_lo),
                                             _mm256_cvtepi16_epi32(e_lo));
            __m256i s_hi = _mm256_add_epi32(_mm256_cvtepi16_epi32(a_hi),
                                             _mm256_cvtepi16_epi32(e_hi));

            s_lo = _mm256_add_epi32(s_lo, _mm256_load_si256((const __m256i*)(_nnL1B + o)));
            s_hi = _mm256_add_epi32(s_hi, _mm256_load_si256((const __m256i*)(_nnL1B + o + 8)));

            s_lo = _mm256_min_epi32(_mm256_max_epi32(s_lo, zero), v255);
            s_hi = _mm256_min_epi32(_mm256_max_epi32(s_hi, zero), v255);

            __m256i p16 = _mm256_packus_epi32(s_lo, s_hi);
            p16 = _mm256_permute4x64_epi64(p16, _MM_SHUFFLE(3,1,2,0));
            __m128i lo16 = _mm256_castsi256_si128(p16);
            __m128i hi16 = _mm256_extracti128_si256(p16, 1);
            _mm_store_si128((__m128i*)(relu1 + o), _mm_packus_epi16(lo16, hi16));
        }
    }
#elif defined(__wasm_simd128__)
    /* WASM SIMD128 ClippedReLU: process 8 int16 → 8 uint8 per iteration.
     *
     * Strategy: widen int16→int32 (4 at a time), add int32 bias, clamp [0,255],
     * then narrow back: i32→i16 (saturating unsigned), then i16→u8 (saturating).
     *
     * Each loop iteration handles 8 int16 inputs → 8 uint8 outputs.
     * NN_L1_OUT=256, so 32 iterations total.
     *
     * WASM narrowing instructions used:
     *   wasm_i16x8_narrow_i32x4 (signed saturating i32→i16)
     *   wasm_u8x16_narrow_i16x8 (unsigned saturating i16→u8, gives uint8)
     */
    {
        for (int o = 0; o < NN_L1_OUT; o += 8) {
            /* Load 8 int16 from acc and ext */
            v128_t a = wasm_v128_load(acc + o);
            v128_t e = wasm_v128_load(ext + o);

            /* Widen lower 4 int16 to int32, add bias, clamp */
            v128_t a_lo32 = wasm_i32x4_extend_low_i16x8(a);
            v128_t e_lo32 = wasm_i32x4_extend_low_i16x8(e);
            v128_t b_lo32 = wasm_v128_load(_nnL1B + o);      /* 4 int32 bias */
            v128_t s_lo = wasm_i32x4_add(wasm_i32x4_add(a_lo32, e_lo32), b_lo32);
            s_lo = wasm_i32x4_min(wasm_i32x4_max(s_lo, wasm_i32x4_splat(0)),
                                   wasm_i32x4_splat(255));

            /* Widen upper 4 int16 to int32, add bias, clamp */
            v128_t a_hi32 = wasm_i32x4_extend_high_i16x8(a);
            v128_t e_hi32 = wasm_i32x4_extend_high_i16x8(e);
            v128_t b_hi32 = wasm_v128_load(_nnL1B + o + 4);  /* next 4 int32 bias */
            v128_t s_hi = wasm_i32x4_add(wasm_i32x4_add(a_hi32, e_hi32), b_hi32);
            s_hi = wasm_i32x4_min(wasm_i32x4_max(s_hi, wasm_i32x4_splat(0)),
                                   wasm_i32x4_splat(255));

            /* Narrow i32→i16 (signed saturating), then i16→u8 (unsigned saturating) */
            v128_t packed16 = wasm_i16x8_narrow_i32x4(s_lo, s_hi);
            /* narrow to u8: we need unsigned saturating — use u8x16_narrow_i16x8.
             * It takes two i16x8 and packs to u8x16. We pass packed16 twice so
             * the lower 8 bytes are exactly our 8 uint8 values. */
            v128_t packed8 = wasm_u8x16_narrow_i16x8(packed16, packed16);
            /* Store only the lower 8 bytes (our 8 results) */
            wasm_v128_store64_lane(relu1 + o, packed8, 0);
        }
    }
#else
    for (int o = 0; o < NN_L1_OUT; o++) {
        int32_t v = (int32_t)acc[o] + (int32_t)ext[o] + _nnL1B[o];
        relu1[o] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
#endif

    /* ── Step 3: L2 matrix-vector multiply ─────────────────────────
     *
     * Weight layout: _nnL2W[o * NN_L2_IN + i] = [64][256]
     * For each output neuron o, compute:
     *   acc2[o] = bias[o] + sum_i(relu1[i] * L2W[o][i])
     *
     * AVX2 FAST KERNEL using maddubs_epi16:
     *   _mm256_maddubs_epi16(a, b):
     *     a = 32 uint8,  b = 32 int8
     *     result = 16 int16: result[j] = a[2j]*b[2j] + a[2j+1]*b[2j+1]  (saturated)
     *     Processes 32 multiply-adds in one instruction!
     *
     *   _mm256_madd_epi16(a, ones):
     *     a = 16 int16, ones = all-ones int16
     *     result = 8 int32: pairwise horizontal add
     *     Completes the int16→int32 reduction.
     *
     * For 256 inputs: 8 AVX2 iterations (256/32=8) per output neuron.
     * Total: 64 outputs × 8 iterations = 512 maddubs instructions.
     * vs old code: 256 iterations × 8 mullo_epi32 = 2048 instructions at
     * ¼ the element throughput. This is the dominant speedup.
     *
     * NOTE: maddubs_epi16 saturates at int16 range. Since relu1 ∈ [0,255]
     * and weights ∈ [-127,127], each product ≤ 255×127=32385 < 32767 (OK).
     * Pairwise sum ≤ 2×32385=64770 which CAN overflow int16 (>32767).
     * FIX: interleave with madd after each maddubs to stay in int32.
     * ──────────────────────────────────────────────────────────────── */
    int32_t acc2[NN_L2_OUT] __attribute__((aligned(32)));

#if defined(__AVXVNNI__)
    /* v3.20: AVX-VNNI path ported from v4.02.  VPDPBUSD performs
     * uint8 activations x int8 weights directly into int32 accumulators,
     * avoiding the maddubs+madd+add sequence and its intermediate int16
     * saturation.  The AVX2 path below remains the portable fallback. */
    {
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = _nnL2W + (o+0) * NN_L2_IN;
            const int8_t *row1 = _nnL2W + (o+1) * NN_L2_IN;
            const int8_t *row2 = _nnL2W + (o+2) * NN_L2_IN;
            const int8_t *row3 = _nnL2W + (o+3) * NN_L2_IN;
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();
            for (int i = 0; i < NN_L2_IN; i += 32) {
                __m256i a = _mm256_load_si256((const __m256i*)(relu1 + i));
                sum0 = _mm256_dpbusd_epi32(sum0, a, _mm256_load_si256((const __m256i*)(row0 + i)));
                sum1 = _mm256_dpbusd_epi32(sum1, a, _mm256_load_si256((const __m256i*)(row1 + i)));
                sum2 = _mm256_dpbusd_epi32(sum2, a, _mm256_load_si256((const __m256i*)(row2 + i)));
                sum3 = _mm256_dpbusd_epi32(sum3, a, _mm256_load_si256((const __m256i*)(row3 + i)));
            }
            acc2[o+0] = _nnL2B[o+0] + _hsum_epi32(sum0);
            acc2[o+1] = _nnL2B[o+1] + _hsum_epi32(sum1);
            acc2[o+2] = _nnL2B[o+2] + _hsum_epi32(sum2);
            acc2[o+3] = _nnL2B[o+3] + _hsum_epi32(sum3);
        }
    }
#elif defined(__AVX2__)
    {
        __m256i ones = _mm256_set1_epi16(1);
        /* 4-way output unroll: share relu1 loads across 4 weight rows.
         * Reduces relu1 memory traffic by 4× and hsum calls from 64 to 16. */
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = _nnL2W + (o+0) * NN_L2_IN;
            const int8_t *row1 = _nnL2W + (o+1) * NN_L2_IN;
            const int8_t *row2 = _nnL2W + (o+2) * NN_L2_IN;
            const int8_t *row3 = _nnL2W + (o+3) * NN_L2_IN;
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();

            for (int i = 0; i < NN_L2_IN; i += 32) {
                __m256i a = _mm256_load_si256((const __m256i*)(relu1 + i));
                __m256i p0 = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row0 + i)));
                __m256i p1 = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row1 + i)));
                __m256i p2 = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row2 + i)));
                __m256i p3 = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row3 + i)));
                sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0, ones));
                sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1, ones));
                sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2, ones));
                sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3, ones));
            }
            acc2[o+0] = _nnL2B[o+0] + _hsum_epi32(sum0);
            acc2[o+1] = _nnL2B[o+1] + _hsum_epi32(sum1);
            acc2[o+2] = _nnL2B[o+2] + _hsum_epi32(sum2);
            acc2[o+3] = _nnL2B[o+3] + _hsum_epi32(sum3);
        }
    }
#elif defined(__wasm_simd128__)
    /* WASM SIMD128 L2 kernel — 4-way output unroll.
     *
     * KEY OPTIMISATION: process 4 output neurons per outer iteration,
     * loading each relu1[i..i+15] block ONCE and using it for all 4 rows.
     * This cuts relu1 memory traffic by 4× — the dominant bottleneck in WASM
     * where the JIT has less cache-aware prefetch than native x86.
     *
     * Each inner iteration (16 relu1 elements):
     *   - 1 load  : relu1[i..i+15]   (shared across 4 outputs)
     *   - 4 loads : row0/1/2/3[i..i+15]
     *   - 8 extend + 8 mul + 8 extadd_pairwise + 4 add  per output
     *
     * vs old 1-wide loop: 4× more loads from relu1 for the same work.
     *
     * NN_L2_OUT=64 is divisible by 4, so no tail needed.
     * NN_L2_IN=256 is divisible by 16, so inner loop has no tail.
     *
     * Numerical identity: 4-wide computes exactly the same dot products as
     * the 1-wide loop — no approximation, no reordering of accumulation.
     * All three builds (AVX2, WASM, scalar) produce identical acc2[] values.
     */
    {
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = _nnL2W + (o+0) * NN_L2_IN;
            const int8_t *row1 = _nnL2W + (o+1) * NN_L2_IN;
            const int8_t *row2 = _nnL2W + (o+2) * NN_L2_IN;
            const int8_t *row3 = _nnL2W + (o+3) * NN_L2_IN;

            v128_t sum0 = wasm_i32x4_splat(0);
            v128_t sum1 = wasm_i32x4_splat(0);
            v128_t sum2 = wasm_i32x4_splat(0);
            v128_t sum3 = wasm_i32x4_splat(0);

            for (int i = 0; i < NN_L2_IN; i += 16) {
                /* Load 16 relu1 uint8 values — shared by all 4 output rows */
                v128_t a8 = wasm_v128_load(relu1 + i);

                /* Zero-extend uint8 activations to int16 once */
                v128_t a_lo = wasm_u16x8_extend_low_u8x16(a8);
                v128_t a_hi = wasm_u16x8_extend_high_u8x16(a8);

/* Helper macro: dot-accumulate one weight row into sumN.
 * Loads 16 int8 weights, sign-extends to int16, multiplies with a_lo/a_hi,
 * pairwise-adds to int32, accumulates into sumN. */
#define _DOT4(sumN, rowN) do {                                          \
    v128_t b8   = wasm_v128_load((rowN) + i);                          \
    v128_t b_lo = wasm_i16x8_extend_low_i8x16(b8);                     \
    v128_t b_hi = wasm_i16x8_extend_high_i8x16(b8);                    \
    (sumN) = wasm_i32x4_add((sumN),                                     \
                 wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_mul(a_lo, b_lo))); \
    (sumN) = wasm_i32x4_add((sumN),                                     \
                 wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_mul(a_hi, b_hi))); \
} while(0)

                _DOT4(sum0, row0);
                _DOT4(sum1, row1);
                _DOT4(sum2, row2);
                _DOT4(sum3, row3);

#undef _DOT4
            }

/* Horizontal reduce 4 int32 lanes → scalar */
#define _HSUM4(v) (wasm_i32x4_extract_lane((v),0) + wasm_i32x4_extract_lane((v),1) \
                 + wasm_i32x4_extract_lane((v),2) + wasm_i32x4_extract_lane((v),3))

            acc2[o+0] = _nnL2B[o+0] + _HSUM4(sum0);
            acc2[o+1] = _nnL2B[o+1] + _HSUM4(sum1);
            acc2[o+2] = _nnL2B[o+2] + _HSUM4(sum2);
            acc2[o+3] = _nnL2B[o+3] + _HSUM4(sum3);

#undef _HSUM4
        }
    }
#else
    for (int o = 0; o < NN_L2_OUT; o++) {
        int32_t s = _nnL2B[o];
        const int8_t *row = _nnL2W + o * NN_L2_IN;
        for (int i = 0; i < NN_L2_IN; i++) s += (int32_t)relu1[i] * row[i];
        acc2[o] = s;
    }
#endif

    /* Step 4: shift + ClippedReLU → relu2[64] uint8 */
    uint8_t relu2[NN_L2_OUT];
    for (int o = 0; o < NN_L2_OUT; o++) {
        int32_t v = acc2[o] >> NN_SHIFT;
        relu2[o] = (uint8_t)(v < 0 ? 0 : v > NN_QB ? NN_QB : v);
    }

    /* Step 5: L3 — integer dot product, single float conversion.
     * Old: 64× (int8 * (float)uint8) accumulated as float.
     * New: 64× (int8 * int32) accumulated as int32, one float at the end.
     * Eliminates 64 integer→float conversion instructions. */
    int32_t l3_sum = 0;
    for (int i = 0; i < NN_L3_IN; i++) l3_sum += (int32_t)_nnL3W[i] * relu2[i];

    /* Step 6: scale + bias → centipawns, clamped to [-2000, +2000] */
    float cp = (float)l3_sum * _nnOutScale + _nnL3B * 320.0f;
    if (cp < -2000.f) cp = -2000.f;
    if (cp >  2000.f) cp =  2000.f;
    return (int)(cp + (cp >= 0.f ? 0.5f : -0.5f));
}

/* ════════════════════════════════════════════════════════════════════
 * nnue_eval_bb — fast eval path for search (uses precomputed Board bb[])
 *
 * Optimisations over nnue_eval:
 *
 * OPT-A  No _build_bb_from_board:
 *   Board already has bb[12] maintained incrementally by board_make/unmake.
 *   We pass them in directly, saving the 64-iteration rebuild loop per eval.
 *
 * OPT-B  2-slot feature cache keyed on (board_hash ^ stm):
 *   Within a single search call _acc_ptr never changes (push/pop not wired),
 *   so the same position is often evaluated with the same board twice
 *   (qsearch stand-pat check + alpha-beta re-eval, or aspiration retry).
 *   A tiny 2-entry direct-mapped cache avoids redundant _compute_extra_feat
 *   + _project_feat_full whenever the position hasn't changed.
 *
 * OPT-C  _compute_extra_feat accepts bb[12] directly:
 *   New _compute_extra_feat_bb variant skips the internal bitboard rebuild.
 *
 * All existing nnue_eval callers (WASM, benchmark) are unaffected.
 * ════════════════════════════════════════════════════════════════════ */

/* Extra-feature computation accepting precomputed bitboards (no bb rebuild). */
static void _compute_extra_feat_bb(float *feat, const uint64_t bb[12], int stm) {
    static const float MAXCNT[6] = {8.f,2.f,2.f,2.f,1.f,1.f};
    static const float MATVAL[6] = {1.f,3.f,3.f,5.f,9.f,0.f};

    if (!_extra_masks_init) _init_extra_masks();

    /* ── Piece counts via popcount ──────────────────────────────── */
    int cnt_w[6], cnt_b[6];
    for (int t=0; t<6; t++) {
        cnt_w[t] = __builtin_popcountll(bb[t]);
        cnt_b[t] = __builtin_popcountll(bb[t+6]);
    }

    int *stm_cnt = (stm==0) ? cnt_w : cnt_b;
    int *opp_cnt = (stm==0) ? cnt_b : cnt_w;
    for (int i=0;i<6;i++) feat[i]   = stm_cnt[i]/MAXCNT[i];
    for (int i=0;i<6;i++) feat[6+i] = opp_cnt[i]/MAXCNT[i];

    float mat=0.f;
    for (int i=0;i<6;i++) mat += (cnt_w[i]+cnt_b[i])*MATVAL[i];
    feat[12] = mat/78.f;
    feat[13] = 1.0f;

    int wk_sq = bb[5]  ? __builtin_ctzll(bb[5])  : -1;
    int bk_sq = bb[11] ? __builtin_ctzll(bb[11]) : -1;

    for (int f=0; f<8; f++) { feat[14+f]=0.f; feat[22+f]=0.f; }

    uint64_t wp = bb[0], bp = bb[6];
    if (stm == 0) {
        uint64_t tmp = wp;
        while (tmp) { int sq=__builtin_ctzll(tmp); tmp&=tmp-1; if(!(_pp_span_w[sq]&bp)) feat[14+(sq&7)]=1.f; }
        tmp = bp;
        while (tmp) { int sq=__builtin_ctzll(tmp); tmp&=tmp-1; if(!(_pp_span_b[sq]&wp)) feat[22+(sq&7)]=1.f; }
    } else {
        uint64_t tmp = bp;
        while (tmp) { int sq=__builtin_ctzll(tmp); tmp&=tmp-1; if(!(_pp_span_b[sq]&wp)) feat[14+(sq&7)]=1.f; }
        tmp = wp;
        while (tmp) { int sq=__builtin_ctzll(tmp); tmp&=tmp-1; if(!(_pp_span_w[sq]&bp)) feat[22+(sq&7)]=1.f; }
    }

    if (wk_sq>=0 && bk_sq>=0) {
        int wf=wk_sq&7, wr=wk_sq>>3, bf=bk_sq&7, br=bk_sq>>3;
        int df=wf>bf?wf-bf:bf-wf, dr=wr>br?wr-br:br-wr;
        feat[30]=(float)(df>dr?df:dr)/7.f;
    } else feat[30]=0.f;
}

/* Per-thread ext cache (v3.13).
 * Uses NnueAccum's cache_key/cache_buf instead of global statics.
 * 4-slot direct-mapped: slot = hash & 3.
 * On hit: reuse projected buffer.  On miss: compute + store. */

int nnue_eval_bb(NnueAccum *na, int stm, const uint8_t *board,
                 const uint64_t bb[12], uint64_t board_hash)
{
    if (!_nnue_ready) return 0;
    if (na->acc_dirty) return 0;  /* v3.14: per-thread dirty flag */

    /* Cache key: mix hash with stm so White/Black perspectives are separate slots */
    uint64_t key = board_hash ^ ((uint64_t)stm * 0x9e3779b97f4a7c15ULL);
    int slot = (int)(key & (EXT_CACHE_SLOTS - 1));

    const int16_t *ext;
    if (na->cache_key[slot] != key) {
        /* Cache miss — compute extra features using precomputed bb[], project */
        float feat[NN_EXTRA];
        _compute_extra_feat_bb(feat, bb, stm);
        int16_t *buf = na->cache_buf[slot];
        memset(buf, 0, NN_L1_OUT * sizeof(int16_t));
        _project_feat_full(buf, feat);
        na->cache_key[slot] = key;
        ext = buf;
    } else {
        /* Cache hit — reuse projected buffer */
        ext = na->cache_buf[slot];
    }

    /* Read HM accumulator from per-thread stack (v3.14) */
    uint8_t relu1[NN_L1_OUT] __attribute__((aligned(32)));
    const int16_t *acc = stm==0 ? na->acc_stack_w[na->acc_ptr] : na->acc_stack_b[na->acc_ptr];

#ifdef __AVX2__
    {
        __m256i v255 = _mm256_set1_epi32(255);
        __m256i zero = _mm256_setzero_si256();
        for (int o = 0; o < NN_L1_OUT; o += 16) {
            __m128i a_lo = _mm_load_si128((const __m128i*)(acc + o));
            __m128i a_hi = _mm_load_si128((const __m128i*)(acc + o + 8));
            __m128i e_lo = _mm_load_si128((const __m128i*)(ext + o));
            __m128i e_hi = _mm_load_si128((const __m128i*)(ext + o + 8));
            __m256i s_lo = _mm256_add_epi32(_mm256_cvtepi16_epi32(a_lo), _mm256_cvtepi16_epi32(e_lo));
            __m256i s_hi = _mm256_add_epi32(_mm256_cvtepi16_epi32(a_hi), _mm256_cvtepi16_epi32(e_hi));
            s_lo = _mm256_add_epi32(s_lo, _mm256_load_si256((const __m256i*)(_nnL1B + o)));
            s_hi = _mm256_add_epi32(s_hi, _mm256_load_si256((const __m256i*)(_nnL1B + o + 8)));
            s_lo = _mm256_min_epi32(_mm256_max_epi32(s_lo, zero), v255);
            s_hi = _mm256_min_epi32(_mm256_max_epi32(s_hi, zero), v255);
            __m256i p16 = _mm256_packus_epi32(s_lo, s_hi);
            p16 = _mm256_permute4x64_epi64(p16, _MM_SHUFFLE(3,1,2,0));
            __m128i lo16 = _mm256_castsi256_si128(p16);
            __m128i hi16 = _mm256_extracti128_si256(p16, 1);
            _mm_store_si128((__m128i*)(relu1 + o), _mm_packus_epi16(lo16, hi16));
        }
    }
#else
    for (int o = 0; o < NN_L1_OUT; o++) {
        int32_t v = (int32_t)acc[o] + (int32_t)ext[o] + _nnL1B[o];
        relu1[o] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
#endif

    /* Steps 3-6: identical to nnue_eval */
    int32_t acc2[NN_L2_OUT] __attribute__((aligned(32)));
#if defined(__AVXVNNI__)
    /* v3.20: AVX-VNNI path ported from v4.02.  VPDPBUSD performs
     * uint8 activations x int8 weights directly into int32 accumulators,
     * avoiding the maddubs+madd+add sequence and its intermediate int16
     * saturation.  The AVX2 path below remains the portable fallback. */
    {
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = _nnL2W + (o+0) * NN_L2_IN;
            const int8_t *row1 = _nnL2W + (o+1) * NN_L2_IN;
            const int8_t *row2 = _nnL2W + (o+2) * NN_L2_IN;
            const int8_t *row3 = _nnL2W + (o+3) * NN_L2_IN;
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();
            for (int i = 0; i < NN_L2_IN; i += 32) {
                __m256i a = _mm256_load_si256((const __m256i*)(relu1 + i));
                sum0 = _mm256_dpbusd_epi32(sum0, a, _mm256_load_si256((const __m256i*)(row0 + i)));
                sum1 = _mm256_dpbusd_epi32(sum1, a, _mm256_load_si256((const __m256i*)(row1 + i)));
                sum2 = _mm256_dpbusd_epi32(sum2, a, _mm256_load_si256((const __m256i*)(row2 + i)));
                sum3 = _mm256_dpbusd_epi32(sum3, a, _mm256_load_si256((const __m256i*)(row3 + i)));
            }
            acc2[o+0] = _nnL2B[o+0] + _hsum_epi32(sum0);
            acc2[o+1] = _nnL2B[o+1] + _hsum_epi32(sum1);
            acc2[o+2] = _nnL2B[o+2] + _hsum_epi32(sum2);
            acc2[o+3] = _nnL2B[o+3] + _hsum_epi32(sum3);
        }
    }
#elif defined(__AVX2__)
    {
        __m256i ones = _mm256_set1_epi16(1);
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = _nnL2W + (o+0) * NN_L2_IN;
            const int8_t *row1 = _nnL2W + (o+1) * NN_L2_IN;
            const int8_t *row2 = _nnL2W + (o+2) * NN_L2_IN;
            const int8_t *row3 = _nnL2W + (o+3) * NN_L2_IN;
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();
            for (int i = 0; i < NN_L2_IN; i += 32) {
                __m256i a  = _mm256_load_si256((const __m256i*)(relu1 + i));
                __m256i p0 = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row0 + i)));
                __m256i p1 = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row1 + i)));
                __m256i p2 = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row2 + i)));
                __m256i p3 = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row3 + i)));
                sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0, ones));
                sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1, ones));
                sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2, ones));
                sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3, ones));
            }
            acc2[o+0] = _nnL2B[o+0] + _hsum_epi32(sum0);
            acc2[o+1] = _nnL2B[o+1] + _hsum_epi32(sum1);
            acc2[o+2] = _nnL2B[o+2] + _hsum_epi32(sum2);
            acc2[o+3] = _nnL2B[o+3] + _hsum_epi32(sum3);
        }
    }
#else
    for (int o = 0; o < NN_L2_OUT; o++) {
        int32_t s = _nnL2B[o];
        const int8_t *row = _nnL2W + o * NN_L2_IN;
        for (int i = 0; i < NN_L2_IN; i++) s += (int32_t)relu1[i] * row[i];
        acc2[o] = s;
    }
#endif

    uint8_t relu2[NN_L2_OUT];
    for (int o = 0; o < NN_L2_OUT; o++) {
        int32_t v = acc2[o] >> NN_SHIFT;
        relu2[o] = (uint8_t)(v < 0 ? 0 : v > NN_QB ? NN_QB : v);
    }
    int32_t l3_sum = 0;
    for (int i = 0; i < NN_L3_IN; i++) l3_sum += (int32_t)_nnL3W[i] * relu2[i];
    float cp = (float)l3_sum * _nnOutScale + _nnL3B * 320.0f;
    if (cp < -2000.f) cp = -2000.f;
    if (cp >  2000.f) cp =  2000.f;
    return (int)(cp + (cp >= 0.f ? 0.5f : -0.5f));
}

/* ── Standalone benchmark (compile with -DNNUE_TEST) ────────────── */
#ifdef NNUE_TEST
#include <assert.h>
#include <time.h>

static long now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000000000L + ts.tv_nsec;
}

int main(int argc, char **argv) {
    const char *path = argc>1 ? argv[1] : "nnue_weights.bin";
    if (nnue_load(path) != 0) return 1;

    /* Allocate a NnueAccum for testing (same as search thread would) */
    NnueAccum test_na __attribute__((aligned(32)));
    memset(&test_na, 0, sizeof(test_na));

    uint8_t board[64] = {
        20,18,19,21,22,19,18,20,
        17,17,17,17,17,17,17,17,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         9, 9, 9, 9, 9, 9, 9, 9,
        12,10,11,13,14,11,10,12,
    };

    nnue_rebuild(&test_na, board);
    int score   = nnue_eval(&test_na, 0, board);
    int score_b = nnue_eval(&test_na, 1, board);
    printf("[NNUE TEST] Starting pos stm=W: %d cp\n", score);
    printf("[NNUE TEST] Starting pos stm=B: %d cp\n", score_b);

    uint8_t board_rd[64]; memcpy(board_rd, board, 64); board_rd[63]=0;
    nnue_rebuild(&test_na, board_rd);
    printf("[NNUE TEST] White rook down stm=W: %d (expect < -200)\n", nnue_eval(&test_na, 0,board_rd));
    printf("[NNUE TEST] White rook down stm=B: %d (expect > +200)\n", nnue_eval(&test_na, 1,board_rd));

    /* Legacy push/pop test (still uses global arrays) */
    nnue_rebuild(&test_na, board);
    NNMove m={52,36,0,0,0};
    nnue_push(board,&m);
    uint8_t board_e4[64]; memcpy(board_e4,board,64);
    board_e4[36]=board_e4[52]; board_e4[52]=0;
    int score_after=nnue_eval(&test_na, 1,board_e4);
    printf("[NNUE TEST] After 1.e4 stm=B: %d\n", score_after);
    nnue_pop();
    int score_back=nnue_eval(&test_na, 0,board);
    printf("[NNUE TEST] After pop (back to start): %d\n", score_back);
    assert(score_back==score);

    /* Throughput benchmark */
    int N=2000000;
    long t0=now_ns();
    volatile int sum=0;
    for(int i=0;i<N;i++) sum+=nnue_eval(&test_na, i&1,board);
    long t1=now_ns();
    double ms=(t1-t0)/1e6;
    printf("[BENCH] nnue_eval x%d: %.1f ms  (%.2f M evals/sec)\n", N, ms, N/ms/1000.0);

    /* Push/pop throughput: simulates search tree — push, eval, pop */
    nnue_rebuild(&test_na, board);
    NNMove moves[4] = {
        {52,36,0,0,0},  /* e2-e4 */
        {12,28,0,0,0},  /* e7-e5 */
        {62,45,0,0,0},  /* g1-f3 */
        {57,42,0,0,0},  /* b8-c6 */
    };
    /* Build post-move boards for each depth level */
    uint8_t boards[5][64];
    memcpy(boards[0], board, 64);
    for (int i=0; i<4; i++) {
        memcpy(boards[i+1], boards[i], 64);
        int f=moves[i].from_sq, to=moves[i].to_sq;
        boards[i+1][to]=boards[i+1][f]; boards[i+1][f]=0;
    }

    int M=500000;
    long t2=now_ns();
    volatile int sum2=0;
    nnue_rebuild(&test_na, boards[0]);
    for (int i=0; i<M; i++) {
        /* Simulate 4-ply push sequence, eval at leaf, pop back */
        for (int d=0; d<4; d++) nnue_push(boards[d], &moves[d]);
        sum2 += nnue_eval(&test_na, (4)&1, boards[4]);
        for (int d=0; d<4; d++) nnue_pop();
    }
    long t3=now_ns();
    double ms2=(t3-t2)/1e6;
    printf("[BENCH] push×4+eval+pop×4 x%d: %.1f ms  (%.2f M ops/sec)\n",
           M, ms2, M/ms2/1000.0);

    /* Search-like pattern: eval at every ply, not just leaf. */
    int S=200000;
    long t4=now_ns();
    volatile int sum3=0;
    nnue_rebuild(&test_na, boards[0]);
    for (int i=0; i<S; i++) {
        sum3 += nnue_eval(&test_na, 0, boards[0]);
        for (int d=0; d<4; d++) {
            nnue_push(boards[d], &moves[d]);
            sum3 += nnue_eval(&test_na, (d+1)&1, boards[d+1]);
        }
        for (int d=0; d<4; d++) nnue_pop();
    }
    long t5=now_ns();
    double ms3=(t5-t4)/1e6;
    printf("[BENCH] eval+push+eval×4+pop×4 x%d (search-like): %.1f ms  (%.2f M evals/sec)\n",
           S, ms3, (S*5)/ms3/1000.0);

    printf("[NNUE TEST] All tests passed.\n");
    return 0;
}
#endif

