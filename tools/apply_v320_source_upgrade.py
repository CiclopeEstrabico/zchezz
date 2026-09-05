#!/usr/bin/env python3
"""Materialize the first v3.20 source upgrade from the v3.14 baseline.

The transform is intentionally narrow:
- rename the UCI engine from 3.14 to 3.20;
- add the AVX-VNNI L2 dot-product path already validated in the v4.02 line;
- leave the v3.14 search policy, NNUE weights, feature set and AVX2 fallback intact.

The script is idempotent and fails loudly if the expected v3.14 source shape changes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NNUE = ROOT / "engine/c/zchezz_v320/nnue.c"
MAIN = ROOT / "engine/c/zchezz_v320/main.c"

vnni_block = r'''#if defined(__AVXVNNI__)
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
'''

needle = "#ifdef __AVX2__\n    {\n        __m256i ones = _mm256_set1_epi16(1);"
replacement = vnni_block + "    {\n        __m256i ones = _mm256_set1_epi16(1);"

text = NNUE.read_text(encoding="utf-8")
if "v3.20: AVX-VNNI path ported from v4.02" not in text:
    count = text.count(needle)
    if count != 2:
        raise SystemExit(f"expected exactly two v3.14 L2 AVX2 kernels, found {count}")
    text = text.replace(needle, replacement, 2)
    NNUE.write_text(text, encoding="utf-8")

main = MAIN.read_text(encoding="utf-8")
if "Zchezz 3.20" not in main:
    if "Zchezz 3.14" not in main:
        raise SystemExit("could not find the v3.14 UCI version string")
    main = main.replace("Zchezz 3.14", "Zchezz 3.20")
    MAIN.write_text(main, encoding="utf-8")

print("v3.20 source upgrade materialized")
