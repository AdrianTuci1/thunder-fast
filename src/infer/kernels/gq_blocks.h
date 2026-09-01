#pragma once
/* gq_blocks.h — quantized block formats (GGUF/ggml compatible) + scalar reference.
 *
 * thunder-fast runs the diffusion loop on CPU. The heavy op is the per-token matmul
 * against quantized weights, which is a de-quantization + dot product. We keep the on-disk
 * block layouts identical to GGUF so a checkpoint converted to GGUF loads as-is, and the
 * scalar routines below are the correctness reference that every SIMD kernel is checked
 * against during validation.
 *
 * NOTE: K-quants (Q4_K/Q5_K/Q6_K) are more intricate (multi-scale + "min" refinement).
 * They are deliberately NOT ported here yet; we start with the whole-family baselines
 * Q4_0 and Q8_0 (see memory §7). Port Q4_K from the exact ggml/GGUF version in use, and
 * cross-check the dequant formula — HF/ggml vary between versions (AGENTS.md).
 */

#include <stdint.h>

#define GQ_QK4_0 32
#define GQ_QK8_0 32

/* ggml fp16 is stored as its IEEE 754 binary16 bit pattern (uint16). */
typedef uint16_t gq_fp16_t;

typedef struct {
    gq_fp16_t d;        /* scale (fp16) */
    uint8_t  qs[GQ_QK4_0 / 2]; /* 4-bit quanta, two per byte (low nibble = even idx) */
} gq_block_q4_0;       /* sizeof == 18 */

typedef struct {
    gq_fp16_t d;        /* scale (fp16) */
    int8_t   qs[GQ_QK8_0];    /* signed 8-bit quanta */
} gq_block_q8_0;       /* sizeof == 34 */

#ifdef __cplusplus
extern "C" {
#endif

/* fp16 bit pattern -> float (correct for normal range; fine for quant scales). */
static inline float gq_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t man  = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        /* subnormal */
        bits = (sign << 31) | (man << 13);
        if (man != 0) {
            /* normalize subnormal */
            int n = 0;
            while (!(man & 0x400u)) { man <<= 1; n++; }
            bits = (sign << 31) | ((uint32_t)(127 - 15 - n) << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = (sign << 31) | 0x7f800000u | (man << 13); /* inf/nan */
    } else {
        bits = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    uint32_t u = bits;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

/* --- dequantize one block into a float row ------------------------------------ */

static inline void gq_dequant_q4_0(const gq_block_q4_0 *b, float *out) {
    const float d = gq_half_to_float(b->d);
    for (int j = 0; j < GQ_QK4_0; j++) {
        const int nib = (b->qs[j >> 1] >> (4 * (j & 1))) & 0x0f;
        out[j] = (float)(nib - 8) * d;
    }
}

static inline void gq_dequant_q8_0(const gq_block_q8_0 *b, float *out) {
    const float d = gq_half_to_float(b->d);
    for (int j = 0; j < GQ_QK8_0; j++) {
        out[j] = (float)b->qs[j] * d;
    }
}

/* --- scalar dot products (the correctness reference) -------------------------- */

/* sum_j q4[j] * q8[j], where q4 and q8 share the same 32-wide block.
 * Returns the *unscaled* sum of (qs4-8)*qs8; caller multiplies by d4*d8.
 * We keep the scale factor out so SIMD paths only accumulate integers. */
static inline int32_t gq_dot_q4_0_q8_0_scal(const gq_block_q4_0 *b4, const gq_block_q8_0 *b8) {
    int32_t acc = 0;
    for (int j = 0; j < GQ_QK4_0; j++) {
        const int nib = (b4->qs[j >> 1] >> (4 * (j & 1))) & 0x0f;
        acc += (nib - 8) * (int32_t)b8->qs[j];
    }
    return acc;
}

static inline int32_t gq_dot_q8_0_q8_0_scal(const gq_block_q8_0 *a, const gq_block_q8_0 *b) {
    int32_t acc = 0;
    for (int j = 0; j < GQ_QK8_0; j++) {
        acc += (int32_t)a->qs[j] * (int32_t)b->qs[j];
    }
    return acc;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
