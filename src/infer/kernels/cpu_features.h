#pragma once
/* cpu_features.h — runtime dispatch of SIMD kernels for thunder-fast.
 *
 * We target the micro-architectures in AGENTS.md / memory §7:
 *   x86:  AVX2 -> AVX-512 -> AVX-512 VNNI  (VNNI = VPDPBUSD int8 dot)
 *   ARM:  NEON -> DotProd (v8.2+) -> I8MM (v8.6+) -> SVE
 *   Apple Silicon: NEON (no SVE on consumer chips); AMX Apple handled elsewhere.
 *
 * All functions are `static inline` and just read the host CPU at call time, so the
 * runtime can pick the best kernel once per process. No external dependency.
 */

#if defined(__x86_64__) || defined(__i386__)
#  define GQ_ARCH_X86 1
#elif defined(__aarch64__) || defined(__arm__)
#  define GQ_ARCH_ARM 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline int gq_has_avx2(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

static inline int gq_has_avx512f(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f");
#else
    return 0;
#endif
}

static inline int gq_has_avx512_vnni(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512vnni");
#else
    return 0;
#endif
}

static inline int gq_has_neon(void) {
#if defined(GQ_ARCH_ARM) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

/* ARMv8.2+ DotProd (vdot/vdotq), ARMv8.6+ I8MM (vmmlaq_s32). Detected via HWCAP. */
#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__))
#  include <sys/auxv.h>          /* getauxval(AT_HWCAP) */
#endif
static inline int gq_has_dotprod(void) {
#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__))
    return (getauxval(AT_HWCAP) & (1UL << 32)) != 0;  /* HWCAP_ASIMDDP */
#else
    return 0;
#endif
}

static inline int gq_has_i8mm(void) {
#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__))
    return (getauxval(AT_HWCAP) & (1UL << 34)) != 0;  /* HWCAP_ASIMDI8MM */
#else
    return 0;
#endif
}

#ifdef __cplusplus
} /* extern "C" */
#endif
