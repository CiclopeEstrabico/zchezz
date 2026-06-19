#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

int main() {
    int16_t vacc[16];
    int16_t vext[16];
    int32_t bias[16];
    uint8_t relu1[16];

    for (int i=0; i<16; i++) {
        vacc[i] = i * 10;
        vext[i] = i * -5;
        bias[i] = 100;
    }

    __m256i va = _mm256_loadu_si256((const __m256i*)vacc);
    __m256i ve = _mm256_loadu_si256((const __m256i*)vext);

    __m256i vsum = _mm256_add_epi16(va, ve);
    __m256i vsum_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vsum, 0));
    __m256i vsum_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vsum, 1));

    __m256i vbias_lo = _mm256_loadu_si256((const __m256i*)(bias));
    __m256i vbias_hi = _mm256_loadu_si256((const __m256i*)(bias + 8));
    
    __m256i s_lo = _mm256_add_epi32(vsum_lo, vbias_lo);
    __m256i s_hi = _mm256_add_epi32(vsum_hi, vbias_hi);

    __m256i zero = _mm256_setzero_si256();
    __m256i v255 = _mm256_set1_epi32(255);
    s_lo = _mm256_max_epi32(s_lo, zero);
    s_lo = _mm256_min_epi32(s_lo, v255);
    s_hi = _mm256_max_epi32(s_hi, zero);
    s_hi = _mm256_min_epi32(s_hi, v255);

    __m256i packed = _mm256_packs_epi32(s_lo, s_hi);
    __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    packed = _mm256_permutevar8x32_epi32(packed, perm_mask);

    __m128i lane0 = _mm256_extracti128_si256(packed, 0);
    __m128i lane1 = _mm256_extracti128_si256(packed, 1);
    __m128i final_8 = _mm_packus_epi16(lane0, lane1);
    
    _mm_storeu_si128((__m128i*)relu1, final_8);

    int sum_avx = 0;
    for (int i=0; i<16; i++) {
        sum_avx += relu1[i];
    }

    int sum_noavx = 0;
    for (int i=0; i<16; i++) {
        int v = vacc[i] + vext[i] + bias[i];
        int r = v < 0 ? 0 : v > 255 ? 255 : v;
        sum_noavx += r;
    }

    printf("AVX sum: %d\nNO-AVX sum: %d\n", sum_avx, sum_noavx);
    for (int i=0; i<16; i++) {
        printf("relu1[%d] = %d\n", i, relu1[i]);
    }

    return 0;
}
