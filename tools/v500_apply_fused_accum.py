#!/usr/bin/env python3
"""Materialize a v5 HalfKP accumulator hot-path candidate.

Fuses the common quiet non-king move from three passes per clean POV
(copy child frame, subtract source feature, add destination feature) into
one AVX2/WASM/scalar pass: child = parent - from_row + to_row.
The transform is modulo-int16 identical to the existing sequence and does
not change features, buckets, search, weights, or quantization.
"""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("--root", default="engine/c/zchezz_v500_fused")
a = p.parse_args()
path = Path(a.root) / "nnue.c"
s = path.read_text(encoding="utf-8")

marker = "/* ── Castle square tables: {king_from, king_to, rook_from, rook_to} ── */"
helper = r'''/* v5 performance candidate: fuse child-frame copy with the two feature
 * deltas for the overwhelmingly common quiet non-king move.  Arithmetic is
 * int16 modular in both implementations, so src-from+to is bit-identical to
 * memcpy; sub(from); add(to), while cutting accumulator memory traffic. */
static inline void _acc_copy_quiet_delta(int16_t *dstW, int16_t *dstB,
                                         const int16_t *srcW, const int16_t *srcB,
                                         const int16_t *L1WT,
                                         uint8_t p, int from, int to,
                                         int bkt_w, int bkt_b,
                                         int updW, int updB) {
    int relW = _piece_rel(p, 1);
    if (relW < 0) return;
    int relB = relW < 5 ? relW + 5 : relW - 5;
    const int16_t *fw = L1WT + (size_t)(bkt_w * NN_FEAT_PER_BUCKET + relW * 64 + (from ^ 56)) * NN_L1_OUT;
    const int16_t *tw = L1WT + (size_t)(bkt_w * NN_FEAT_PER_BUCKET + relW * 64 + (to   ^ 56)) * NN_L1_OUT;
    const int16_t *fb = L1WT + (size_t)(bkt_b * NN_FEAT_PER_BUCKET + relB * 64 +  from        ) * NN_L1_OUT;
    const int16_t *tb = L1WT + (size_t)(bkt_b * NN_FEAT_PER_BUCKET + relB * 64 +  to          ) * NN_L1_OUT;
#ifdef __AVX2__
    if (updW) { _mm_prefetch((const char*)fw, _MM_HINT_T0); _mm_prefetch((const char*)tw, _MM_HINT_T0); }
    if (updB) { _mm_prefetch((const char*)fb, _MM_HINT_T0); _mm_prefetch((const char*)tb, _MM_HINT_T0); }
    for (int o = 0; o < NN_L1_OUT; o += 16) {
        if (updW) {
            __m256i v = _mm256_load_si256((const __m256i*)(srcW + o));
            v = _mm256_sub_epi16(v, _mm256_load_si256((const __m256i*)(fw + o)));
            v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i*)(tw + o)));
            _mm256_store_si256((__m256i*)(dstW + o), v);
        }
        if (updB) {
            __m256i v = _mm256_load_si256((const __m256i*)(srcB + o));
            v = _mm256_sub_epi16(v, _mm256_load_si256((const __m256i*)(fb + o)));
            v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i*)(tb + o)));
            _mm256_store_si256((__m256i*)(dstB + o), v);
        }
    }
#elif defined(__wasm_simd128__)
    for (int o = 0; o < NN_L1_OUT; o += 8) {
        if (updW) {
            v128_t v = wasm_v128_load(srcW + o);
            v = wasm_i16x8_sub(v, wasm_v128_load(fw + o));
            v = wasm_i16x8_add(v, wasm_v128_load(tw + o));
            wasm_v128_store(dstW + o, v);
        }
        if (updB) {
            v128_t v = wasm_v128_load(srcB + o);
            v = wasm_i16x8_sub(v, wasm_v128_load(fb + o));
            v = wasm_i16x8_add(v, wasm_v128_load(tb + o));
            wasm_v128_store(dstB + o, v);
        }
    }
#else
    if (updW) for (int o = 0; o < NN_L1_OUT; o++) dstW[o] = (int16_t)(srcW[o] - fw[o] + tw[o]);
    if (updB) for (int o = 0; o < NN_L1_OUT; o++) dstB[o] = (int16_t)(srcB[o] - fb[o] + tb[o]);
#endif
}

'''
if s.count(marker) != 1:
    raise SystemExit("castle marker mismatch")
s = s.replace(marker, helper + marker, 1)

old = '''    int updW = !dw, updB = !db;
    if (updW) memcpy(na->acc_stack_w[dst], na->acc_stack_w[src], NN_L1_OUT * sizeof(int16_t));
    if (updB) memcpy(na->acc_stack_b[dst], na->acc_stack_b[src], NN_L1_OUT * sizeof(int16_t));
    int16_t *cW = na->acc_stack_w[dst], *cB = na->acc_stack_b[dst];

    if (m->castle) {'''
new = '''    int updW = !dw, updB = !db;
    int16_t *cW = na->acc_stack_w[dst], *cB = na->acc_stack_b[dst];
    int f0 = m->from_sq, to0 = m->to_sq;
    uint8_t p0 = board[f0], cap0 = board[to0];
    int fused_quiet = !m->castle && !cap0 && !m->is_epc && !m->prom && PC_TYPE(p0) != PT_KING;
    if (fused_quiet) {
        _acc_copy_quiet_delta(cW, cB,
                              na->acc_stack_w[src], na->acc_stack_b[src],
                              L1WT, p0, f0, to0, bw, bb, updW, updB);
    } else {
        if (updW) memcpy(cW, na->acc_stack_w[src], NN_L1_OUT * sizeof(int16_t));
        if (updB) memcpy(cB, na->acc_stack_b[src], NN_L1_OUT * sizeof(int16_t));
    }

    if (m->castle) {'''
if s.count(old) != 1:
    raise SystemExit("push prologue mismatch")
s = s.replace(old, new, 1)

old = '''        } else {
            _acc_sub_piece(cW, cB, L1WT, p, f, bw, bb, updW, updB);
            if (cap) _acc_sub_piece(cW, cB, L1WT, cap, to, bw, bb, updW, updB);
            if (m->is_epc) {
                int epsq = (PC_COLOR(p) == COL_W) ? to + 8 : to - 8;
                if (board[epsq]) _acc_sub_piece(cW, cB, L1WT, board[epsq], epsq, bw, bb, updW, updB);
            }
            uint8_t landing = m->prom ? (uint8_t)(PC_COLOR(p) | m->prom) : p;
            _acc_add_piece(cW, cB, L1WT, landing, to, bw, bb, updW, updB);
        }'''
new = '''        } else if (!fused_quiet) {
            _acc_sub_piece(cW, cB, L1WT, p, f, bw, bb, updW, updB);
            if (cap) _acc_sub_piece(cW, cB, L1WT, cap, to, bw, bb, updW, updB);
            if (m->is_epc) {
                int epsq = (PC_COLOR(p) == COL_W) ? to + 8 : to - 8;
                if (board[epsq]) _acc_sub_piece(cW, cB, L1WT, board[epsq], epsq, bw, bb, updW, updB);
            }
            uint8_t landing = m->prom ? (uint8_t)(PC_COLOR(p) | m->prom) : p;
            _acc_add_piece(cW, cB, L1WT, landing, to, bw, bb, updW, updB);
        }'''
if s.count(old) != 1:
    raise SystemExit("quiet delta block mismatch")
s = s.replace(old, new, 1)

path.write_text(s, encoding="utf-8")
print("applied fused quiet accumulator candidate")
