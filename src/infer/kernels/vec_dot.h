#pragma once
/* vec_dot.h — runtime dispatch for dequant+dot kernels.
 * Picks the best SIMD variant the current CPU supports, falling back to the scalar
 * reference (gq_blocks.h). Callers multiply the returned int32 by d_a * d_b to get the
 * final fp dot product. All SIMD paths are unverified here — validate vs scalar on build.
 */

#include "gq_blocks.h"
#include "cpu_features.h"

#if defined(GQ_ARCH_X86)
#  include "vec_dot_x86_avx2.h"
#  include "vec_dot_x86_vnni.h"
#elif defined(GQ_ARCH_ARM)
#  include "vec_dot_arm_neon.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline int32_t gq_dot_q4_0_q8_0(const gq_block_q4_0 *a, const gq_block_q8_0 *b) {
#if defined(GQ_ARCH_X86)
    if (gq_has_avx512_vnni()) return gq_avx512vnni_dot_q4_0_q8_0(a, b);
    if (gq_has_avx2())        return gq_avx2_dot_q4_0_q8_0(a, b);
#elif defined(GQ_ARCH_ARM)
    if (gq_has_neon())        return gq_neon_dot_q4_0_q8_0(a, b);
#endif
    return gq_dot_q4_0_q8_0_scal(a, b);
}

static inline int32_t gq_dot_q8_0_q8_0(const gq_block_q8_0 *a, const gq_block_q8_0 *b) {
#if defined(GQ_ARCH_X86)
    if (gq_has_avx2())        return gq_avx2_dot_q8_0_q8_0(a, b);
#elif defined(GQ_ARCH_ARM)
    if (gq_has_neon())        return gq_neon_dot_q8_0_q8_0(a, b);
#endif
    return gq_dot_q8_0_q8_0_scal(a, b);
}

#ifdef __cplusplus
} /* extern "C" */
#endif
