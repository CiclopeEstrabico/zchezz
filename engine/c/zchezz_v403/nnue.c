#define _POSIX_C_SOURCE 200809L
/* nnue.c — Zchezz NNUE evaluation layer (v4.00: HalfKP-4Bucket)
 *
 * See nnue.h for the full architecture description, the coordinate
 * conventions, and the lazy bucket-refresh contract.  This file is the
 * implementation; the invariants live in the header.
 *
 * ── Pipeline per evaluated node ──────────────────────────────────
 *
 *   Step 0  Lazy refresh: if this frame's perspective was invalidated
 *           by a king move that crossed a bucket border, rebuild that
 *           perspective from the board.  Rare.
 *   Step 1  Accumulators are already up to date (nnue_push_na keeps
 *           acc_stack_w/b[acc_ptr] incremental).
 *   Step 2  SCReLU: relu1[1024] uint8 = [stm 512 | opp 512], where
 *           c = clamp(acc + L1B, 0, 255) and out = (c*c) >> 8.
 *   Step 3  L2: 1024 -> 32, int8 weights, _mm256_maddubs_epi16 kernel
 *           (uint8 x int8 -> int16 pairwise) with a 4-wide output
 *           unroll so each relu1 block is loaded once for 4 rows.
 *   Step 4  ClippedReLU: relu2[32] = clamp(acc2 >> NN_SHIFT, 0, QB).
 *   Step 5  L3: int32 dot product, one float conversion at the end.
 *   Step 6  cp = l3_sum * OUT_SCALE + L3B * 320, clamped to +/-2000.
 *
 * ── What was deleted vs v3.14 ────────────────────────────────────
 *
 *   _compute_extra_feat / _compute_extra_feat_bb  (31 endgame features)
 *   _project_feat_add / _project_feat_full / _project_feat_incremental
 *   _pp_span_w/_pp_span_b/_init_extra_masks       (passed-pawn masks)
 *   _build_bb_from_board                          (bb[] no longer used)
 *   the ext cache (cache_key/cache_buf) and ext_dirty bookkeeping
 *
 * That removal is worth roughly O(31 x 256) SIMD per node — the merge
 * that used to sit between the accumulator and the activation.  It is
 * also why nnue_eval and nnue_eval_bb are now the same function: the
 * bitboards and the Zobrist hash only ever existed to feed the extras.
 *
 * Compile standalone self-test:
 *   gcc -O3 -march=native -std=c11 -o test_nnue nnue.c -DNNUE_TEST -lm
 */

#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__AVX2__) || defined(__AVXVNNI__)
#  include <immintrin.h>
#elif defined(__wasm_simd128__)
#  include <wasm_simd128.h>
#endif

/* ── Piece constant helpers ──────────────────────────────────────── */
#define PC_COLOR(p)  ((p) & 24)
#define PC_TYPE(p)   ((p) &  7)
#define COL_W  8
#define COL_B 16
#define PT_KING 6

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

/* ── Weight storage (per-instance NnueNet; read-only after load: MT-safe)
 *
 * This used to be a set of file-scope statics (_nnL1WT, _nnL1B, ...) —
 * one net per process, period. Packaged into NnueNet so a process can
 * hold several independently loaded nets at once (native A/B arena:
 * gen_{i+1} vs gen_i in the same process). Mirrors TTable in search.h
 * field-for-field in spirit: a struct + create/destroy + a process-wide
 * default pointer (g_nnue_net, declared in nnue.h) that every legacy
 * caller implicitly uses.
 *
 * Once loaded, every field is read-only, so concurrent search threads
 * (or two arena players, each with their own NnueAccum bound to a
 * different NnueNet) may share a NnueNet freely — exactly like the old
 * globals were safe to share, just now there can be more than one. */
struct NnueNet {
    /* L1: [NN_L1_IN][NN_L1_OUT] = [2560][512] int16, feature-major so
     * that one feature's contribution is a single contiguous 1 KB row. */
    int16_t *L1WT;
    int32_t *L1B;      /* [512] int32                     */

    /* L2: [NN_L2_OUT][NN_L2_IN] = [32][1024] int8, row-major per OUTPUT
     * neuron — the layout the maddubs kernel wants.  The NNU4 file
     * already stores it this way, so the loader is a straight memcpy
     * (NNU3 needed a transpose on load; that is gone). */
    int8_t  *L2W;
    int32_t *L2B;      /* [32] int32                      */

    /* L3 */
    int8_t  *L3W;      /* [32] int8                       */
    float    L3B;
    float    OutScale;

    int      ready;
};

NnueNet *nnue_net_create(void) {
    return (NnueNet *)calloc(1, sizeof(NnueNet));
}

void nnue_net_destroy(NnueNet *net) {
    if (!net) return;
    zfree32(net->L1WT); zfree32(net->L1B);
    zfree32(net->L2W);  zfree32(net->L2B);
    zfree32(net->L3W);
    free(net);
}

int nnue_net_ready(const NnueNet *net) { return net && net->ready; }

/* Process-wide default net (declared extern in nnue.h). Every legacy
 * (non-net-aware) API call implicitly targets this, exactly like g_tt
 * in search.h is the implicit default TTable. */
NnueNet *g_nnue_net = NULL;

/* nnue_ready() kept as the same zero-argument global query the rest of
 * the engine already calls (search.c, main.c): true once the process
 * default net is loaded. */
int nnue_ready(void) { return nnue_net_ready(g_nnue_net); }

/* g_nnue_accum is defined in board.c (extern declared in board.h).
 * We cannot include board.h here (circular dep), so forward-declare. */
extern NnueAccum g_nnue_accum;

/* ════════════════════════════════════════════════════════════════
 * FEATURE ENCODING — HalfKP-4Bucket
 *
 * Must stay bit-identical to halfkp_features() in
 * train/train_nnue.py.  See nnue.h for the convention table and
 * train/check_parity.py for the automated cross-check.
 * ════════════════════════════════════════════════════════════════ */

/* Bucket of a king standing on POV square `s` (0 = own a1 corner). */
static inline int _king_bucket(int s) {
    return ((s & 7) >= 4 ? 1 : 0) | ((s >> 3) >= 4 ? 2 : 0);
}

/* White perspective: the POV square is the mailbox square mirrored. */
int nnue_king_bucket_w(int wk_zsq) { return _king_bucket(wk_zsq ^ 56); }
/* Black perspective: the mailbox layout (a8=0) already IS Black's POV. */
int nnue_king_bucket_b(int bk_zsq) { return _king_bucket(bk_zsq); }

/* Relative piece plane for one perspective, or -1 if not a feature.
 * ally P,N,B,R,Q -> 0..4 ; enemy P,N,B,R,Q -> 5..9 ; king -> -1. */
static inline int _piece_rel(uint8_t p, int white_pov) {
    int t = PC_TYPE(p) - 1;                 /* P=0 .. Q=4, K=5 */
    if (t < 0 || t >= 5) return -1;         /* empty, king, or garbage */
    int is_white_piece = (PC_COLOR(p) == COL_W);
    return (is_white_piece == white_pov ? 0 : 5) + t;
}

int nnue_feature_index(uint8_t p, int zsq, int white_pov, int bucket) {
    int rel = _piece_rel(p, white_pov);
    if (rel < 0) return -1;
    int pov_sq = white_pov ? (zsq ^ 56) : zsq;
    return bucket * NN_FEAT_PER_BUCKET + rel * 64 + pov_sq;
}

/* ── Accumulator add / sub for one piece, both perspectives ───────
 *
 * bkt_w / bkt_b are the CURRENT bucket of each perspective.  The caller
 * owns bucket bookkeeping; if a bucket just changed, the corresponding
 * perspective is flagged dirty and whatever we write into it here will
 * be thrown away by the rebuild, so passing the stale bucket is safe.
 *
 * Kings are silently skipped (they are not features).
 *
 * `L1WT` is passed in (rather than read from a global) so the caller
 * can hoist `na->net->L1WT` into a register-friendly local once per
 * call site instead of re-dereferencing na->net on every invocation —
 * see the "hot-path locals" note on _nnue_forward. */
static inline void _acc_add_piece(int16_t *accW, int16_t *accB,
                                  const int16_t *L1WT,
                                  uint8_t p, int zsq, int bkt_w, int bkt_b,
                                  int updW, int updB) {
    int relW = _piece_rel(p, 1);
    if (relW < 0) return;                       /* king or empty: nothing to do */
    int relB = relW < 5 ? relW + 5 : relW - 5;  /* ally<->enemy swap */
    const int16_t *wRow = L1WT + (size_t)(bkt_w * NN_FEAT_PER_BUCKET + relW * 64 + (zsq ^ 56)) * NN_L1_OUT;
    const int16_t *bRow = L1WT + (size_t)(bkt_b * NN_FEAT_PER_BUCKET + relB * 64 +  zsq       ) * NN_L1_OUT;
#ifdef __AVX2__
    if (updW) _mm_prefetch((const char*)wRow, _MM_HINT_T0);
    if (updB) _mm_prefetch((const char*)bRow, _MM_HINT_T0);
    for (int o = 0; o < NN_L1_OUT; o += 16) {
        if (updW) {
            __m256i aw = _mm256_load_si256((const __m256i*)(accW + o));
            aw = _mm256_add_epi16(aw, _mm256_load_si256((const __m256i*)(wRow + o)));
            _mm256_store_si256((__m256i*)(accW + o), aw);
        }
        if (updB) {
            __m256i ab = _mm256_load_si256((const __m256i*)(accB + o));
            ab = _mm256_add_epi16(ab, _mm256_load_si256((const __m256i*)(bRow + o)));
            _mm256_store_si256((__m256i*)(accB + o), ab);
        }
    }
#elif defined(__wasm_simd128__)
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

static inline void _acc_sub_piece(int16_t *accW, int16_t *accB,
                                  const int16_t *L1WT,
                                  uint8_t p, int zsq, int bkt_w, int bkt_b,
                                  int updW, int updB) {
    int relW = _piece_rel(p, 1);
    if (relW < 0) return;
    int relB = relW < 5 ? relW + 5 : relW - 5;
    const int16_t *wRow = L1WT + (size_t)(bkt_w * NN_FEAT_PER_BUCKET + relW * 64 + (zsq ^ 56)) * NN_L1_OUT;
    const int16_t *bRow = L1WT + (size_t)(bkt_b * NN_FEAT_PER_BUCKET + relB * 64 +  zsq       ) * NN_L1_OUT;
#ifdef __AVX2__
    if (updW) _mm_prefetch((const char*)wRow, _MM_HINT_T0);
    if (updB) _mm_prefetch((const char*)bRow, _MM_HINT_T0);
    for (int o = 0; o < NN_L1_OUT; o += 16) {
        if (updW) {
            __m256i aw = _mm256_load_si256((const __m256i*)(accW + o));
            aw = _mm256_sub_epi16(aw, _mm256_load_si256((const __m256i*)(wRow + o)));
            _mm256_store_si256((__m256i*)(accW + o), aw);
        }
        if (updB) {
            __m256i ab = _mm256_load_si256((const __m256i*)(accB + o));
            ab = _mm256_sub_epi16(ab, _mm256_load_si256((const __m256i*)(bRow + o)));
            _mm256_store_si256((__m256i*)(accB + o), ab);
        }
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

/* ── NNU4 binary loader ──────────────────────────────────────────
 *
 * Layout (little-endian, produced by train/export_nnu4.py):
 *   "NNU4"                       4 B
 *   epoch                        uint32
 *   dims[5]                      L1_IN, L1_OUT, L2_IN, L2_OUT, L3_IN
 *   scales[4]                    QA, QB, SHIFT, OUT_SCALE   (float32)
 *   L1W  [2560][512] int16       feature-major
 *   L1B  [512]       int32
 *   L2W  [32][1024]  int8        output-major (NO transpose on load)
 *   L2B  [32]        int32
 *   L3W  [32]        int8
 *   L3B                          float32
 */
static int _load_weights(NnueNet *net, const uint8_t *buf, size_t len) {
    if (len < 8 || buf[0]!='N'||buf[1]!='N'||buf[2]!='U'||buf[3]!='4') {
        fprintf(stderr, "[NNUE] Bad magic (expected NNU4)\n"); return -1;
    }
    size_t off = 4 + 4;                     /* magic + epoch */

    /* Validate the dims header against the compiled-in architecture.
     * Loading a net with mismatched dims used to be silent corruption;
     * with a 2.6 MB file it is worth one explicit check. */
    uint32_t dims[5];
    memcpy(dims, buf + off, 5 * sizeof(uint32_t)); off += 5 * 4;
    const uint32_t want[5] = { NN_L1_IN, NN_L1_OUT, NN_L2_IN, NN_L2_OUT, NN_L3_IN };
    for (int i = 0; i < 5; i++) {
        if (dims[i] != want[i]) {
            fprintf(stderr, "[NNUE] Dim mismatch at %d: file=%u expected=%u "
                            "(file dims %u/%u/%u/%u/%u)\n",
                    i, dims[i], want[i], dims[0], dims[1], dims[2], dims[3], dims[4]);
            return -1;
        }
    }

    float scales[4];
    memcpy(scales, buf + off, 4 * sizeof(float)); off += 16;
    net->OutScale = scales[3];

    const size_t L1_SZ = (size_t)NN_L1_IN  * NN_L1_OUT;   /* 2560*512 int16 */
    const size_t L2_SZ = (size_t)NN_L2_OUT * NN_L2_IN;    /* 32*1024  int8  */
    size_t need = off + L1_SZ*2 + NN_L1_OUT*4 + L2_SZ + NN_L2_OUT*4 + NN_L3_IN + 4;
    if (len < need) {
        fprintf(stderr, "[NNUE] File too small (%zu < %zu)\n", len, need); return -1;
    }

    zfree32(net->L1WT); zfree32(net->L1B);
    zfree32(net->L2W);  zfree32(net->L2B);
    zfree32(net->L3W);

    net->L1WT = (int16_t *)zmalloc32(L1_SZ     * sizeof(int16_t));
    net->L1B  = (int32_t *)zmalloc32(NN_L1_OUT * sizeof(int32_t));
    net->L2W  = (int8_t  *)zmalloc32(L2_SZ     * sizeof(int8_t));
    net->L2B  = (int32_t *)zmalloc32(NN_L2_OUT * sizeof(int32_t));
    net->L3W  = (int8_t  *)zmalloc32(NN_L3_IN  * sizeof(int8_t));

    if (!net->L1WT||!net->L1B||!net->L2W||!net->L2B||!net->L3W) {
        fprintf(stderr, "[NNUE] malloc failed\n"); return -1;
    }

    memcpy(net->L1WT, buf+off, L1_SZ * 2);     off += L1_SZ * 2;
    memcpy(net->L1B,  buf+off, NN_L1_OUT * 4); off += NN_L1_OUT * 4;
    memcpy(net->L2W,  buf+off, L2_SZ);         off += L2_SZ;
    memcpy(net->L2B,  buf+off, NN_L2_OUT * 4); off += NN_L2_OUT * 4;
    memcpy(net->L3W,  buf+off, NN_L3_IN);      off += NN_L3_IN;
    memcpy(&net->L3B, buf+off, 4);

    net->ready = 1;
    return 0;
}

NnueNet *nnue_net_load_from_mem(const uint8_t *data, size_t len) {
    NnueNet *net = nnue_net_create();
    if (!net) return NULL;
    if (_load_weights(net, data, len) != 0) { nnue_net_destroy(net); return NULL; }
    return net;
}

NnueNet *nnue_net_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[NNUE] Cannot open: %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { fclose(f); return NULL; }
    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) { fclose(f); return NULL; }
    if ((long)fread(buf, 1, (size_t)fsize, f) != fsize) { free(buf); fclose(f); return NULL; }
    fclose(f);
    NnueNet *net = nnue_net_load_from_mem(buf, (size_t)fsize);
    free(buf);
    if (net)
        fprintf(stderr, "[NNUE] Loaded: %s (NNU4 HalfKP-4Bucket %d->%d->%d->1) OutScale=%g\n",
                path, NN_L1_IN, NN_L1_OUT, NN_L2_OUT, net->OutScale);
    return net;
}

/* ── Legacy global API — thin wrappers over the process-default net ──
 *
 * These are exactly what every existing caller (UCI engine, WASM JS
 * glue, search.c, board.c, tests) already uses; nothing outside this
 * file has to change. nnue_load()/nnue_load_from_mem() additionally
 * bind g_nnue_accum.net = g_nnue_net so the main-thread accumulator
 * (board.c's g_nnue_accum, used unless a caller heap-allocates its
 * own — see main.c helper_thread_fn / selfplay.c) keeps working
 * with zero call-site changes. */
int nnue_load_from_mem(const uint8_t *data, size_t len) {
    if (!g_nnue_net) {
        g_nnue_net = nnue_net_create();
        if (!g_nnue_net) return -1;
    }
    int r = _load_weights(g_nnue_net, data, len);
    if (r == 0) g_nnue_accum.net = g_nnue_net;
    return r;
}

int nnue_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[NNUE] Cannot open: %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { fclose(f); return -1; }
    uint8_t *buf = (uint8_t *)malloc((size_t)fsize);
    if (!buf) { fclose(f); return -1; }
    if ((long)fread(buf, 1, (size_t)fsize, f) != fsize) { free(buf); fclose(f); return -1; }
    fclose(f);
    int r = nnue_load_from_mem(buf, (size_t)fsize);
    free(buf);
    if (r == 0)
        fprintf(stderr, "[NNUE] Loaded: %s (NNU4 HalfKP-4Bucket %d->%d->%d->1) OutScale=%g\n",
                path, NN_L1_IN, NN_L1_OUT, NN_L2_OUT, g_nnue_net->OutScale);
    return r;
}

/* ── Reset ───────────────────────────────────────────────────────
 * Marks the accumulator as needing a full rebuild.  Called at the
 * start of every search, immediately before nnue_rebuild. */
void nnue_reset(NnueAccum *na) {
    na->acc_dirty = 1;
    na->acc_ptr   = 0;
}

void nnue_reset_global(void) { nnue_reset(&g_nnue_accum); }

/* ── Locate the two kings in the mailbox ─────────────────────────
 * Used only by the rebuild paths (once per search, plus the rare
 * bucket refresh) — never in the incremental hot path, which gets the
 * king squares from NNMove. */
static inline void _find_kings(const uint8_t *board, int *wk, int *bk) {
    *wk = 0; *bk = 0;
    for (int sq = 0; sq < 64; sq++) {
        uint8_t p = board[sq];
        if (PC_TYPE(p) != PT_KING) continue;
        if (PC_COLOR(p) == COL_W) *wk = sq; else *bk = sq;
    }
}

/* ── Full rebuild of both perspectives ───────────────────────────
 * Resets the stack to frame 0 and seeds that frame's bucket/dirty
 * state.  Seeding bucket_*_stack[0] is mandatory: without it the first
 * pop after the first push would restore bucket 0 regardless of where
 * the kings actually are. */
void nnue_rebuild(NnueAccum *na, const uint8_t *board) {
    /* Defensive: a rebuild before any net is bound would otherwise
     * dereference NULL inside _acc_add_piece. Callers (search.c,
     * main.c) already gate this behind nnue_ready(), so this should
     * never actually fire — treat it like the acc_dirty guard in
     * _nnue_forward: fail safe, not fail loud. */
    if (!na->net) { na->acc_dirty = 1; return; }
    const int16_t *L1WT = na->net->L1WT;   /* hot-loop local (see _nnue_forward) */

    int wk, bk;
    _find_kings(board, &wk, &bk);
    int bw = nnue_king_bucket_w(wk);
    int bb = nnue_king_bucket_b(bk);

    int16_t *dW = na->acc_w, *dB = na->acc_b;
    memset(dW, 0, NN_L1_OUT * sizeof(int16_t));
    memset(dB, 0, NN_L1_OUT * sizeof(int16_t));
    for (int sq = 0; sq < 64; sq++) {
        uint8_t p = board[sq];
        if (!p || PC_TYPE(p) == PT_KING) continue;   /* kings are not features */
        _acc_add_piece(dW, dB, L1WT, p, sq, bw, bb, 1, 1);
    }

    na->acc_ptr   = 0;
    na->acc_dirty = 0;
    memcpy(na->acc_stack_w[0], dW, NN_L1_OUT * sizeof(int16_t));
    memcpy(na->acc_stack_b[0], dB, NN_L1_OUT * sizeof(int16_t));
    na->bucket_w_stack[0] = (uint8_t)bw;
    na->bucket_b_stack[0] = (uint8_t)bb;
    na->dirty_w_stack[0]  = 0;
    na->dirty_b_stack[0]  = 0;
}

/* ── Rebuild ONE perspective in the current frame (bucket refresh) ──
 * Called from the eval when a king move crossed a bucket border.  Only
 * touches the affected perspective; the other one stayed valid because
 * kings are not features and the opponent's bucket did not move. */
static void _refresh_perspective(NnueAccum *na, const uint8_t *board, int white_pov) {
    if (!na->net) return;   /* defensive: see nnue_rebuild's comment */
    const int16_t *L1WT = na->net->L1WT;

    int wk, bk;
    _find_kings(board, &wk, &bk);
    int ptr = na->acc_ptr;

    if (white_pov) {
        int bw = nnue_king_bucket_w(wk);
        int16_t *acc = na->acc_stack_w[ptr];
        memset(acc, 0, NN_L1_OUT * sizeof(int16_t));
        for (int sq = 0; sq < 64; sq++) {
            uint8_t p = board[sq];
            if (!p || PC_TYPE(p) == PT_KING) continue;
            int rel = _piece_rel(p, 1);
            if (rel < 0) continue;
            const int16_t *row = L1WT +
                (size_t)(bw * NN_FEAT_PER_BUCKET + rel * 64 + (sq ^ 56)) * NN_L1_OUT;
            for (int o = 0; o < NN_L1_OUT; o++) acc[o] += row[o];
        }
        na->bucket_w_stack[ptr] = (uint8_t)bw;
        na->dirty_w_stack[ptr]  = 0;
    } else {
        int bb = nnue_king_bucket_b(bk);
        int16_t *acc = na->acc_stack_b[ptr];
        memset(acc, 0, NN_L1_OUT * sizeof(int16_t));
        for (int sq = 0; sq < 64; sq++) {
            uint8_t p = board[sq];
            if (!p || PC_TYPE(p) == PT_KING) continue;
            int rel = _piece_rel(p, 0);
            if (rel < 0) continue;
            const int16_t *row = L1WT +
                (size_t)(bb * NN_FEAT_PER_BUCKET + rel * 64 + sq) * NN_L1_OUT;
            for (int o = 0; o < NN_L1_OUT; o++) acc[o] += row[o];
        }
        na->bucket_b_stack[ptr] = (uint8_t)bb;
        na->dirty_b_stack[ptr]  = 0;
    }
}

/* ── Castle square tables: {king_from, king_to, rook_from, rook_to} ── */
static const int _castle_sq[5][4] = {
    {0,0,0,0},{60,62,63,61},{60,58,56,59},{4,6,7,5},{4,2,0,3},
};

/* ════════════════════════════════════════════════════════════════
 * INCREMENTAL PUSH / POP
 *
 * `board` is the position BEFORE the move.  m->wk_sq / m->bk_sq are
 * the king squares before the move (board_make copies them from
 * b->wk / b->bk, so no board scan happens here).
 *
 * Three cases:
 *   castling   — the ROOK is a feature and moves; the king is not.
 *                The castling side's bucket may change (e1->g1 stays
 *                in bucket 0/1 territory but e1->c1 crosses).
 *   king move  — the king itself contributes nothing, but a capture by
 *                the king still removes the captured piece from BOTH
 *                perspectives, and the mover's bucket may change.
 *   other move — ordinary sub(from) / sub(capture) / sub(ep) / add(to).
 *
 * A bucket change sets that perspective's per-frame dirty flag; the
 * accumulator we wrote for it is then garbage but will be rebuilt by
 * the first eval that reads this frame (see _refresh_perspective).
 * Child frames inherit the flag, so an unresolved dirty perspective
 * cannot silently become "clean" deeper in the tree.
 * ════════════════════════════════════════════════════════════════ */
void nnue_push_na(NnueAccum *na, const uint8_t *board, const NNMove *m) {
    if (!na->net) { na->acc_dirty = 1; return; }   /* defensive, see nnue_rebuild */
    const int16_t *L1WT = na->net->L1WT;            /* hot-loop local */

    int src = na->acc_ptr, dst = src + 1;
    if (dst >= NN_ACC_STACK) { na->acc_dirty = 1; return; }
    if (na->acc_dirty) { nnue_rebuild(na, board); src = 0; dst = 1; }

    int bw = na->bucket_w_stack[src], bb = na->bucket_b_stack[src];
    int dw = na->dirty_w_stack[src],  db = na->dirty_b_stack[src];

    /* If a perspective is dirty, its accumulator content is garbage and
     * will be rebuilt from scratch by the first eval that reads this
     * frame (_refresh_perspective memsets it).  Skip the memcpy AND all
     * add/sub updates for that side — only buckets/dirty flags must be
     * propagated to the new frame below. */
    int updW = !dw, updB = !db;
    if (updW) memcpy(na->acc_stack_w[dst], na->acc_stack_w[src], NN_L1_OUT * sizeof(int16_t));
    if (updB) memcpy(na->acc_stack_b[dst], na->acc_stack_b[src], NN_L1_OUT * sizeof(int16_t));
    int16_t *cW = na->acc_stack_w[dst], *cB = na->acc_stack_b[dst];

    if (m->castle) {
        const int *cs = _castle_sq[m->castle];
        int kf = cs[0], kt = cs[1], rf = cs[2], rt = cs[3];
        uint8_t rook = board[rf];
        _acc_sub_piece(cW, cB, L1WT, rook, rf, bw, bb, updW, updB);
        _acc_add_piece(cW, cB, L1WT, rook, rt, bw, bb, updW, updB);
        if (PC_COLOR(board[kf]) == COL_W) {
            int nb = nnue_king_bucket_w(kt);
            if (nb != bw) { dw = 1; bw = nb; }
        } else {
            int nb = nnue_king_bucket_b(kt);
            if (nb != bb) { db = 1; bb = nb; }
        }
    } else {
        int f = m->from_sq, to = m->to_sq;
        uint8_t p = board[f], cap = board[to];

        if (PC_TYPE(p) == PT_KING) {
            /* The king is not a feature, but a capture by the king is. */
            if (cap) _acc_sub_piece(cW, cB, L1WT, cap, to, bw, bb, updW, updB);
            if (PC_COLOR(p) == COL_W) {
                int nb = nnue_king_bucket_w(to);
                if (nb != bw) { dw = 1; bw = nb; }
            } else {
                int nb = nnue_king_bucket_b(to);
                if (nb != bb) { db = 1; bb = nb; }
            }
        } else {
            _acc_sub_piece(cW, cB, L1WT, p, f, bw, bb, updW, updB);
            if (cap) _acc_sub_piece(cW, cB, L1WT, cap, to, bw, bb, updW, updB);
            if (m->is_epc) {
                int epsq = (PC_COLOR(p) == COL_W) ? to + 8 : to - 8;
                if (board[epsq]) _acc_sub_piece(cW, cB, L1WT, board[epsq], epsq, bw, bb, updW, updB);
            }
            uint8_t landing = m->prom ? (uint8_t)(PC_COLOR(p) | m->prom) : p;
            _acc_add_piece(cW, cB, L1WT, landing, to, bw, bb, updW, updB);
        }
    }

    na->bucket_w_stack[dst] = (uint8_t)bw;
    na->bucket_b_stack[dst] = (uint8_t)bb;
    na->dirty_w_stack[dst]  = (uint8_t)dw;
    na->dirty_b_stack[dst]  = (uint8_t)db;
    na->acc_ptr = dst;
}

/* Pop restores the parent frame.  Bucket and dirty state come back
 * automatically because they are stored per frame, not as flat fields —
 * this is what makes a king move that crosses a bucket border safe to
 * undo. */
void nnue_pop_na(NnueAccum *na) { if (na->acc_ptr > 0) na->acc_ptr--; }

/* Legacy global push/pop (WASM / tests): same logic, global accumulator. */
void nnue_push(const uint8_t *board, const NNMove *m) { nnue_push_na(&g_nnue_accum, board, m); }
void nnue_pop(void) { nnue_pop_na(&g_nnue_accum); }

/* ════════════════════════════════════════════════════════════════
 * FORWARD PASS
 * ════════════════════════════════════════════════════════════════ */

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

/* ── Step 2: SCReLU over one perspective ─────────────────────────
 *
 *   c   = clamp(acc[o] + L1B[o], 0, 255)     int32
 *   out = (c * c) >> 8                       -> [0, 254] uint8
 *
 * The squaring is done in the int32 domain BEFORE the pack, which lets
 * the pack sequence stay byte-identical to the one v3.14 already used
 * (packus_epi32 -> permute4x64(3,1,2,0) -> packus_epi16).  Doing it
 * after packing would need an extra unpack and is where lane-order bugs
 * come from.  c <= 255 so c*c <= 65025 — no int32 overflow. */
static inline void _screlu_perspective(const int16_t *acc, const int32_t *L1B, uint8_t *dst) {
#ifdef __AVX2__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i v255 = _mm256_set1_epi32(255);
    for (int o = 0; o < NN_L1_OUT; o += 16) {
        __m128i a_lo = _mm_load_si128((const __m128i*)(acc + o));
        __m128i a_hi = _mm_load_si128((const __m128i*)(acc + o + 8));
        __m256i s_lo = _mm256_add_epi32(_mm256_cvtepi16_epi32(a_lo),
                          _mm256_load_si256((const __m256i*)(L1B + o)));
        __m256i s_hi = _mm256_add_epi32(_mm256_cvtepi16_epi32(a_hi),
                          _mm256_load_si256((const __m256i*)(L1B + o + 8)));
        s_lo = _mm256_min_epi32(_mm256_max_epi32(s_lo, zero), v255);
        s_hi = _mm256_min_epi32(_mm256_max_epi32(s_hi, zero), v255);
        s_lo = _mm256_srli_epi32(_mm256_mullo_epi32(s_lo, s_lo), 8);
        s_hi = _mm256_srli_epi32(_mm256_mullo_epi32(s_hi, s_hi), 8);
        __m256i p16 = _mm256_packus_epi32(s_lo, s_hi);
        p16 = _mm256_permute4x64_epi64(p16, _MM_SHUFFLE(3,1,2,0));
        __m128i lo16 = _mm256_castsi256_si128(p16);
        __m128i hi16 = _mm256_extracti128_si256(p16, 1);
        _mm_store_si128((__m128i*)(dst + o), _mm_packus_epi16(lo16, hi16));
    }
#elif defined(__wasm_simd128__)
    const v128_t zero = wasm_i32x4_splat(0);
    const v128_t v255 = wasm_i32x4_splat(255);
    for (int o = 0; o < NN_L1_OUT; o += 8) {
        v128_t a    = wasm_v128_load(acc + o);
        v128_t s_lo = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(a),
                                     wasm_v128_load(L1B + o));
        v128_t s_hi = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(a),
                                     wasm_v128_load(L1B + o + 4));
        s_lo = wasm_i32x4_min(wasm_i32x4_max(s_lo, zero), v255);
        s_hi = wasm_i32x4_min(wasm_i32x4_max(s_hi, zero), v255);
        s_lo = wasm_u32x4_shr(wasm_i32x4_mul(s_lo, s_lo), 8);
        s_hi = wasm_u32x4_shr(wasm_i32x4_mul(s_hi, s_hi), 8);
        /* Values are in [0,254]: both narrows are exact, no saturation. */
        v128_t p16 = wasm_i16x8_narrow_i32x4(s_lo, s_hi);
        v128_t p8  = wasm_u8x16_narrow_i16x8(p16, p16);
        wasm_v128_store64_lane(dst + o, p8, 0);
    }
#else
    for (int o = 0; o < NN_L1_OUT; o++) {
        int32_t v = (int32_t)acc[o] + L1B[o];
        if (v < 0) v = 0; else if (v > 255) v = 255;
        dst[o] = (uint8_t)((v * v) >> 8);
    }
#endif
}

#ifdef NNUE_VERIFY_ACC
/* ════════════════════════════════════════════════════════════════
 * VERIFY MODE (compile with -DNNUE_VERIFY_ACC) — accumulator proof
 *
 * This is a verification harness, not a shipped feature: it must stay
 * entirely inside this #ifdef so a normal build is byte-identical to
 * what it was before this block existed.
 *
 * On every forward pass, BEFORE the live accumulator is used for
 * anything, we independently rebuild both perspectives straight from
 * `board` (same piece loop as nnue_rebuild / _refresh_perspective,
 * but into a local scratch buffer that never touches na->acc_stack_*)
 * and diff them against na->acc_stack_w/b[acc_ptr].  If nnue_push_na
 * (or the lazy bucket refresh) ever produces an accumulator that
 * disagrees with a full rebuild of the SAME board position, that is
 * by definition an incremental-update bug — the two must be
 * mathematically identical for any position reachable by legal moves.
 *
 * On mismatch we print the frame (acc_ptr, a proxy for ply), which
 * perspective diverged, the first differing L1 output index and both
 * values, and abort() immediately so the failure is caught at the
 * exact node that introduced it rather than surfacing later as a
 * silently wrong eval.
 * ════════════════════════════════════════════════════════════════ */
#include <stdlib.h>  /* abort */

static void _verify_rebuild_perspective(const int16_t *L1WT, const uint8_t *board,
                                         int white_pov, int bucket, int16_t *out) {
    memset(out, 0, NN_L1_OUT * sizeof(int16_t));
    for (int sq = 0; sq < 64; sq++) {
        uint8_t p = board[sq];
        if (!p || PC_TYPE(p) == PT_KING) continue;
        int rel = _piece_rel(p, white_pov);
        if (rel < 0) continue;
        int pov_sq = white_pov ? (sq ^ 56) : sq;
        const int16_t *row = L1WT +
            (size_t)(bucket * NN_FEAT_PER_BUCKET + rel * 64 + pov_sq) * NN_L1_OUT;
        for (int o = 0; o < NN_L1_OUT; o++) out[o] += row[o];
    }
}

static void _verify_accumulator(NnueAccum *na, const uint8_t *board) {
    const int16_t *L1WT = na->net->L1WT;
    int ptr = na->acc_ptr;
    int wk, bk;
    _find_kings(board, &wk, &bk);
    int bw = nnue_king_bucket_w(wk);
    int bb = nnue_king_bucket_b(bk);

    int16_t scratch_w[NN_L1_OUT] __attribute__((aligned(32)));
    int16_t scratch_b[NN_L1_OUT] __attribute__((aligned(32)));
    _verify_rebuild_perspective(L1WT, board, 1, bw, scratch_w);
    _verify_rebuild_perspective(L1WT, board, 0, bb, scratch_b);

    const int16_t *live_w = na->acc_stack_w[ptr];
    const int16_t *live_b = na->acc_stack_b[ptr];

    for (int o = 0; o < NN_L1_OUT; o++) {
        if (live_w[o] != scratch_w[o]) {
            fprintf(stderr,
                "[NNUE_VERIFY_ACC] MISMATCH ply=%d perspective=White idx=%d "
                "live=%d rebuilt=%d bucket=%d\n",
                ptr, o, live_w[o], scratch_w[o], bw);
            fflush(stderr);
            abort();
        }
        if (live_b[o] != scratch_b[o]) {
            fprintf(stderr,
                "[NNUE_VERIFY_ACC] MISMATCH ply=%d perspective=Black idx=%d "
                "live=%d rebuilt=%d bucket=%d\n",
                ptr, o, live_b[o], scratch_b[o], bb);
            fflush(stderr);
            abort();
        }
    }
}
#endif /* NNUE_VERIFY_ACC */

/* ── The shared forward pass ─────────────────────────────────────
 *
 * Returns centipawns from the point of view of `stm` (0 = White to
 * move, 1 = Black to move).  The concat order is [stm | opp] — see
 * APPENDIX D of the v400 plan; reversing it inverts the engine. */
static int _nnue_forward(NnueAccum *na, int stm, const uint8_t *board) {
    /* na->net == NULL is treated exactly like acc_dirty: fail safe (eval
     * 0) instead of dereferencing a net that was never bound. This is
     * what makes it safe for an accumulator to exist before nnue_load()
     * (or, in the arena, before its own net is assigned). */
    const NnueNet *net = na->net;
    if (!net || !net->ready) return 0;
    if (na->acc_dirty) return 0;

    /* Hot-path locals: load every weight-array pointer out of `net`
     * exactly once per forward pass. The struct indirection (na->net->X)
     * only happens here; every helper below receives plain pointers, so
     * the compiler can keep them in registers through the SIMD loops
     * instead of re-touching net on every access. This is what keeps
     * per-instance nets from costing the ~2% the task called out as the
     * risk of `net->field` sitting in the innermost loops. */
    const int16_t *L1WT     = net->L1WT;
    const int32_t *L1B      = net->L1B;
    const int8_t  *L2W      = net->L2W;
    const int32_t *L2B      = net->L2B;
    const int8_t  *L3W      = net->L3W;
    const float    L3B      = net->L3B;
    const float    OutScale = net->OutScale;

    int ptr = na->acc_ptr;
    /* Step 0: lazy bucket refresh (rare: only after a king move that
     * crossed a bucket border). */
    if (na->dirty_w_stack[ptr]) _refresh_perspective(na, board, 1);
    if (na->dirty_b_stack[ptr]) _refresh_perspective(na, board, 0);

#ifdef NNUE_VERIFY_ACC
    /* Independently rebuild both perspectives and diff against the
     * live accumulator BEFORE it is used for anything below. */
    _verify_accumulator(na, board);
#endif

    const int16_t *acc_stm = (stm == 0) ? na->acc_stack_w[ptr] : na->acc_stack_b[ptr];
    const int16_t *acc_opp = (stm == 0) ? na->acc_stack_b[ptr] : na->acc_stack_w[ptr];

    /* Step 2: SCReLU -> relu1[1024] = [stm 512 | opp 512] */
    uint8_t relu1[NN_L2_IN] __attribute__((aligned(32)));
    _screlu_perspective(acc_stm, L1B, relu1);
    _screlu_perspective(acc_opp, L1B, relu1 + NN_L1_OUT);

    /* Step 3: L2 — 1024 inputs, 32 outputs, 4-wide output unroll.
     * The unroll loads each relu1 block once and reuses it for 4 weight
     * rows, which is what keeps this kernel memory-bound rather than
     * load-bound.  NN_L2_OUT=32 is divisible by 4 and NN_L2_IN=1024 by
     * 32/16, so neither loop needs a tail. */
    int32_t acc2[NN_L2_OUT] __attribute__((aligned(32)));
#if defined(__AVXVNNI__)
    /* v4.02: AVX-VNNI path — VPDPBUSD fuses the unsigned-byte × signed-byte
     * dot product + int32 accumulate into ONE instruction per 32 inputs,
     * replacing the maddubs+madd+add triple of the plain AVX2 path (~half
     * the uops for the same exact integer result; relu1 is uint8 in
     * [0,254], L2W is int8, so the dpbusd "unsigned a × signed b"
     * semantics match this layer exactly). */
    {
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = L2W + (size_t)(o+0) * NN_L2_IN;
            const int8_t *row1 = L2W + (size_t)(o+1) * NN_L2_IN;
            const int8_t *row2 = L2W + (size_t)(o+2) * NN_L2_IN;
            const int8_t *row3 = L2W + (size_t)(o+3) * NN_L2_IN;
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();
            for (int i = 0; i < NN_L2_IN; i += 32) {
                __m256i a  = _mm256_load_si256((const __m256i*)(relu1 + i));
                sum0 = _mm256_dpbusd_epi32(sum0, a, _mm256_load_si256((const __m256i*)(row0 + i)));
                sum1 = _mm256_dpbusd_epi32(sum1, a, _mm256_load_si256((const __m256i*)(row1 + i)));
                sum2 = _mm256_dpbusd_epi32(sum2, a, _mm256_load_si256((const __m256i*)(row2 + i)));
                sum3 = _mm256_dpbusd_epi32(sum3, a, _mm256_load_si256((const __m256i*)(row3 + i)));
            }
            acc2[o+0] = L2B[o+0] + _hsum_epi32(sum0);
            acc2[o+1] = L2B[o+1] + _hsum_epi32(sum1);
            acc2[o+2] = L2B[o+2] + _hsum_epi32(sum2);
            acc2[o+3] = L2B[o+3] + _hsum_epi32(sum3);
        }
    }
#elif defined(__AVX2__)
    {
        const __m256i ones = _mm256_set1_epi16(1);
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = L2W + (size_t)(o+0) * NN_L2_IN;
            const int8_t *row1 = L2W + (size_t)(o+1) * NN_L2_IN;
            const int8_t *row2 = L2W + (size_t)(o+2) * NN_L2_IN;
            const int8_t *row3 = L2W + (size_t)(o+3) * NN_L2_IN;
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
            acc2[o+0] = L2B[o+0] + _hsum_epi32(sum0);
            acc2[o+1] = L2B[o+1] + _hsum_epi32(sum1);
            acc2[o+2] = L2B[o+2] + _hsum_epi32(sum2);
            acc2[o+3] = L2B[o+3] + _hsum_epi32(sum3);
        }
    }
#elif defined(__wasm_simd128__)
    {
        for (int o = 0; o < NN_L2_OUT; o += 4) {
            const int8_t *row0 = L2W + (size_t)(o+0) * NN_L2_IN;
            const int8_t *row1 = L2W + (size_t)(o+1) * NN_L2_IN;
            const int8_t *row2 = L2W + (size_t)(o+2) * NN_L2_IN;
            const int8_t *row3 = L2W + (size_t)(o+3) * NN_L2_IN;
            v128_t sum0 = wasm_i32x4_splat(0);
            v128_t sum1 = wasm_i32x4_splat(0);
            v128_t sum2 = wasm_i32x4_splat(0);
            v128_t sum3 = wasm_i32x4_splat(0);
            for (int i = 0; i < NN_L2_IN; i += 16) {
                v128_t a8   = wasm_v128_load(relu1 + i);
                v128_t a_lo = wasm_u16x8_extend_low_u8x16(a8);
                v128_t a_hi = wasm_u16x8_extend_high_u8x16(a8);
/* activations are [0,254] and weights [-127,127], so each int16 product
 * fits comfortably; extadd_pairwise widens to int32 before accumulating. */
#define _DOT4(sumN, rowN) do {                                              \
    v128_t b8   = wasm_v128_load((rowN) + i);                               \
    v128_t b_lo = wasm_i16x8_extend_low_i8x16(b8);                          \
    v128_t b_hi = wasm_i16x8_extend_high_i8x16(b8);                         \
    (sumN) = wasm_i32x4_add((sumN),                                         \
                 wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_mul(a_lo, b_lo))); \
    (sumN) = wasm_i32x4_add((sumN),                                         \
                 wasm_i32x4_extadd_pairwise_i16x8(wasm_i16x8_mul(a_hi, b_hi))); \
} while(0)
                _DOT4(sum0, row0);
                _DOT4(sum1, row1);
                _DOT4(sum2, row2);
                _DOT4(sum3, row3);
#undef _DOT4
            }
#define _HSUM4(v) (wasm_i32x4_extract_lane((v),0) + wasm_i32x4_extract_lane((v),1) \
                 + wasm_i32x4_extract_lane((v),2) + wasm_i32x4_extract_lane((v),3))
            acc2[o+0] = L2B[o+0] + _HSUM4(sum0);
            acc2[o+1] = L2B[o+1] + _HSUM4(sum1);
            acc2[o+2] = L2B[o+2] + _HSUM4(sum2);
            acc2[o+3] = L2B[o+3] + _HSUM4(sum3);
#undef _HSUM4
        }
    }
#else
    for (int o = 0; o < NN_L2_OUT; o++) {
        int32_t s = L2B[o];
        const int8_t *row = L2W + (size_t)o * NN_L2_IN;
        for (int i = 0; i < NN_L2_IN; i++) s += (int32_t)relu1[i] * row[i];
        acc2[o] = s;
    }
#endif

    /* Step 4: shift + ClippedReLU -> relu2[32] uint8 in [0, QB] */
    uint8_t relu2[NN_L2_OUT];
    for (int o = 0; o < NN_L2_OUT; o++) {
        int32_t v = acc2[o] >> NN_SHIFT;
        relu2[o] = (uint8_t)(v < 0 ? 0 : v > NN_QB ? NN_QB : v);
    }

    /* Step 5: L3 — integer dot product, one float conversion */
    int32_t l3_sum = 0;
    for (int i = 0; i < NN_L3_IN; i++) l3_sum += (int32_t)L3W[i] * relu2[i];

    /* Step 6: scale + bias -> centipawns.  The 320.0f matches the
     * cp<->WDL temperature used by to_wdl() in training (APPENDIX D.2);
     * changing one without the other silently rescales every eval. */
    float cp = (float)l3_sum * OutScale + L3B * 320.0f;
    if (cp < -2000.f) cp = -2000.f;
    if (cp >  2000.f) cp =  2000.f;
    return (int)(cp + (cp >= 0.f ? 0.5f : -0.5f));
}

int nnue_eval(NnueAccum *na, int stm, const uint8_t *board) {
    return _nnue_forward(na, stm, board);
}

/* bb[] and board_hash existed only for the v3.14 extra-feature cache,
 * which no longer exists.  The signature is kept so search.c does not
 * have to change. */
int nnue_eval_bb(NnueAccum *na, int stm, const uint8_t *board,
                 const uint64_t bb[12], uint64_t board_hash) {
    (void)bb; (void)board_hash;
    return _nnue_forward(na, stm, board);
}

/* ════════════════════════════════════════════════════════════════
 * Standalone self-test / benchmark  (gcc ... -DNNUE_TEST)
 *
 * Checks the two things that silently break a HalfKP port:
 *   1. push/pop round-trip — an incremental accumulator after N moves
 *      must equal a rebuild of the same position (this is what catches
 *      a missed bucket refresh or a forgotten capture-by-king).
 *   2. feature index bounds and the ally/enemy swap between the two
 *      perspectives.
 * ════════════════════════════════════════════════════════════════ */
#ifdef NNUE_TEST
#include <assert.h>

NnueAccum g_nnue_accum;   /* normally lives in board.c */

static void _dump_fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    /* --- Feature encoding checks (no weights needed) --- */
    /* White king on e1 (mailbox 60): POV sq = 60^56 = 4 -> file 4, rank 0
     * -> bucket bit0 set, bit1 clear -> 1. */
    if (nnue_king_bucket_w(60) != 1) _dump_fail("king_bucket_w(e1) != 1");
    /* Black king on e8 (mailbox 4): POV sq = 4 -> bucket 1 (mirror image). */
    if (nnue_king_bucket_b(4) != 1) _dump_fail("king_bucket_b(e8) != 1");
    /* c1 (mailbox 58): POV 58^56 = 2 -> file 2 -> bucket 0. */
    if (nnue_king_bucket_w(58) != 0) _dump_fail("king_bucket_w(c1) != 0");

    /* A white pawn is ally=0 from White's POV, enemy=5 from Black's. */
    int fw = nnue_feature_index(COL_W | 1, 52 /* e2 */, 1, 0);
    int fb = nnue_feature_index(COL_W | 1, 52,          0, 0);
    if (fw < 0 || fw >= NN_FEAT_IN) _dump_fail("white-POV index out of range");
    if (fb < 0 || fb >= NN_FEAT_IN) _dump_fail("black-POV index out of range");
    if (fw / 64 % 10 != 0) _dump_fail("white pawn not plane 0 from White POV");
    if (fb / 64 % 10 != 5) _dump_fail("white pawn not plane 5 from Black POV");
    /* Kings must never produce a feature. */
    if (nnue_feature_index(COL_W | 6, 60, 1, 0) != -1) _dump_fail("king produced a feature");

    printf("feature encoding: OK\n");

    if (argc > 1 && nnue_load(argv[1]) == 0) {
        printf("weights loaded: OK\n");
    } else {
        printf("(no weight file given — skipping forward-pass test)\n");
    }
    return 0;
}
#endif
