#!/usr/bin/env python3
"""Widen the AVX2 L2 output unroll from 4 to 8 outputs.

This is an arithmetic-identity optimization: it reuses each 32-byte relu1
load across eight output rows instead of four.  Integer accumulation order
within each output is unchanged, so scores must remain bit-identical.
"""
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='engine/c/zchezz_v500_l2x8')
a = ap.parse_args()
p = Path(a.root) / 'nnue.c'
s = p.read_text(encoding='utf-8')

old = '''#elif defined(__AVX2__)
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
'''
new = '''#elif defined(__AVX2__)
    {
        const __m256i ones = _mm256_set1_epi16(1);
        /* Eight output accumulators + activation + temporary fit in the 16
         * architectural YMM registers.  Each relu1 cache line is now loaded
         * four times for the 32 outputs instead of eight. */
        for (int o = 0; o < NN_L2_OUT; o += 8) {
            const int8_t *row0 = L2W + (size_t)(o+0) * NN_L2_IN;
            const int8_t *row1 = L2W + (size_t)(o+1) * NN_L2_IN;
            const int8_t *row2 = L2W + (size_t)(o+2) * NN_L2_IN;
            const int8_t *row3 = L2W + (size_t)(o+3) * NN_L2_IN;
            const int8_t *row4 = L2W + (size_t)(o+4) * NN_L2_IN;
            const int8_t *row5 = L2W + (size_t)(o+5) * NN_L2_IN;
            const int8_t *row6 = L2W + (size_t)(o+6) * NN_L2_IN;
            const int8_t *row7 = L2W + (size_t)(o+7) * NN_L2_IN;
            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();
            __m256i sum4 = _mm256_setzero_si256();
            __m256i sum5 = _mm256_setzero_si256();
            __m256i sum6 = _mm256_setzero_si256();
            __m256i sum7 = _mm256_setzero_si256();
            for (int i = 0; i < NN_L2_IN; i += 32) {
                __m256i a = _mm256_load_si256((const __m256i*)(relu1 + i));
                __m256i p;
                p = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row0 + i)));
                sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p, ones));
                p = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row1 + i)));
                sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p, ones));
                p = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row2 + i)));
                sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p, ones));
                p = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row3 + i)));
                sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p, ones));
                p = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row4 + i)));
                sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(p, ones));
                p = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row5 + i)));
                sum5 = _mm256_add_epi32(sum5, _mm256_madd_epi16(p, ones));
                p = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row6 + i)));
                sum6 = _mm256_add_epi32(sum6, _mm256_madd_epi16(p, ones));
                p = _mm256_maddubs_epi16(a, _mm256_load_si256((const __m256i*)(row7 + i)));
                sum7 = _mm256_add_epi32(sum7, _mm256_madd_epi16(p, ones));
            }
            acc2[o+0] = L2B[o+0] + _hsum_epi32(sum0);
            acc2[o+1] = L2B[o+1] + _hsum_epi32(sum1);
            acc2[o+2] = L2B[o+2] + _hsum_epi32(sum2);
            acc2[o+3] = L2B[o+3] + _hsum_epi32(sum3);
            acc2[o+4] = L2B[o+4] + _hsum_epi32(sum4);
            acc2[o+5] = L2B[o+5] + _hsum_epi32(sum5);
            acc2[o+6] = L2B[o+6] + _hsum_epi32(sum6);
            acc2[o+7] = L2B[o+7] + _hsum_epi32(sum7);
        }
    }
'''
if s.count(old) != 1:
    raise SystemExit(f'expected one AVX2 L2 block, found {s.count(old)}')
p.write_text(s.replace(old, new), encoding='utf-8')
print('applied AVX2 L2 x8 output unroll')
