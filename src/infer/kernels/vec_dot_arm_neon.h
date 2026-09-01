#pragma once
/* vec_dot_arm_neon.h — NEON dot kernels (int16 madd via vmull). Unverified; check scalar ref. */
#if defined(GQ_ARCH_ARM) && defined(__ARM_NEON)
#include <arm_neon.h>

/* 8 Q4_0 bytes -> 16 int8 in [-8,7] (el 2i = low nibble, 2i+1 = high nibble). */
static inline int8x16_t gq_neon_q4_s8(const uint8_t *q) {
    uint8x8_t nib = vld1_u8(q);
    uint8x8_t lo  = vand_u8(nib, vdup_n_u8(0x0f));
    uint8x8_t hi  = vand_u8(vshr_n_u8(nib, 4), vdup_n_u8(0x0f));
    uint8x8x2_t z = vzip_u8(lo, hi);
    int8x16_t qq  = vreinterpretq_s8_u8(vcombine_u8(z.val[0], z.val[1]));
    return vsubq_s8(qq, vdupq_n_s8(8));
}

static inline int32_t gq_neon_reduce(int32x4_t acc) {
    int32x2_t v = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    v = vpadd_s32(v, v);
    return vget_lane_s32(vpadd_s32(v, v), 0);
}

static inline int32_t gq_neon_dot_q4_0_q8_0(const gq_block_q4_0 *a, const gq_block_q8_0 *b) {
    int32x4_t acc = vmovq_n_s32(0);
    for (int h = 0; h < 2; h++) {
        int8x16_t q4 = gq_neon_q4_s8(a->qs + h * 8);
        int8x16_t q8 = vld1q_s8(b->qs + h * 16);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(q4),  vget_low_s8(q8)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(q4), vget_high_s8(q8)));
    }
    return gq_neon_reduce(acc);
}

static inline int32_t gq_neon_dot_q8_0_q8_0(const gq_block_q8_0 *a, const gq_block_q8_0 *b) {
    int32x4_t acc = vmovq_n_s32(0);
    for (int h = 0; h < 2; h++) {
        int8x16_t x = vld1q_s8(a->qs + h * 16);
        int8x16_t y = vld1q_s8(b->qs + h * 16);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x),  vget_low_s8(y)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(x), vget_high_s8(y)));
    }
    return gq_neon_reduce(acc);
}
#endif /* GQ_ARCH_ARM && __ARM_NEON */
