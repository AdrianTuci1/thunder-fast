#pragma once
/* vec_dot_x86_avx2.h — AVX2 dot kernels (int16 madd). Unverified; check vs scalar ref. */
#if defined(GQ_ARCH_X86)
#include <immintrin.h>

/* 8 Q4_0 bytes -> 16 int16 in [-8,7] (el 2i = low nibble, 2i+1 = high nibble). */
static inline __m256i gq_avx2_q4_s16(const uint8_t *q) {
    __m128i nib = _mm_loadl_epi64((const __m128i*)q);
    __m128i lo  = _mm_and_si128(nib, _mm_set1_epi8(0x0f));
    __m128i hi  = _mm_and_si128(_mm_srli_epi16(nib, 4), _mm_set1_epi8(0x0f));
    __m128i qq  = _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
    return _mm256_cvtepi8_epi16(qq);
}

static inline int32_t gq_avx2_reduce(__m256i acc) {
    __m128i v = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xee));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x11));
    return _mm_cvtsi128_si32(v);
}

__attribute__((target("avx2")))
static inline int32_t gq_avx2_dot_q4_0_q8_0(const gq_block_q4_0 *a, const gq_block_q8_0 *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int h = 0; h < 2; h++) {
        __m256i q4 = gq_avx2_q4_s16(a->qs + h * 8);
        __m256i q8 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b->qs + h * 16)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(q4, q8));
    }
    return gq_avx2_reduce(acc);
}

__attribute__((target("avx2")))
static inline int32_t gq_avx2_dot_q8_0_q8_0(const gq_block_q8_0 *a, const gq_block_q8_0 *b) {
    __m256i acc = _mm256_setzero_si256();
    for (int h = 0; h < 2; h++) {
        __m256i x = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a->qs + h * 16)));
        __m256i y = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b->qs + h * 16)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
    }
    return gq_avx2_reduce(acc);
}
#endif /* GQ_ARCH_X86 */
