#pragma once
/* vec_dot_x86_vnni.h — AVX-512 VNNI (VPDPBUSD) dot for Q4_0 x Q8_0.
 * dpbusd sums unsigned a * signed b per 4-byte group. We feed a = q4 nibbles as u8 (0..15)
 * and b = q8 as s8; since q4_actual = q4_u8 - 8, we subtract 8*sum(q8) at the end.
 * NOTE: unverified here — validate against gq_dot_q4_0_q8_0_scal on the build MCU.
 */
#if defined(GQ_ARCH_X86)
#include <immintrin.h>

__attribute__((target("avx512f,avx512vnni")))
static inline int32_t gq_avx512vnni_dot_q4_0_q8_0(const gq_block_q4_0 *a, const gq_block_q8_0 *b) {
    /* Pack 32 nibbles into 32 u8 (element j = byte j/2 nibble j%2); zero the upper 32 bytes. */
    uint8_t q4u[64] = {0};
    for (int i = 0; i < 16; i++) {
        q4u[2 * i]     = a->qs[i] & 0x0f;
        q4u[2 * i + 1] = a->qs[i] >> 4;
    }
    uint8_t ones[64] = {0};
    for (int i = 0; i < 32; i++) ones[i] = 1;

    __m512i v  = _mm512_loadu_si512((const void*)q4u);
    __m512i q8 = _mm512_maskz_loadu_epi8((__mmask64)0xFFFFFFFFULL, b->qs);
    __m512i on = _mm512_loadu_si512((const void*)ones);

    int32_t dp  = _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), v, q8));
    int32_t sq8 = _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), on, q8));
    return dp - 8 * sq8;
}
#endif /* GQ_ARCH_X86 */
