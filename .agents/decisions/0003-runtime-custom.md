# ADR 0003 — Runtime: binar custom, construit peste kernel-urile ggml

**Status:** Accepted
**Data:** 2026-08-31
**Rezolvă:** open question #4 din memory.md

## Context
Modelul nostru este un **diffusion LM** (difuzie continuă, atenție bidirecțională, denoising pe
24 de pași). llama.cpp (aplicația completă) e construit pentru **generare autoregresivă**:
mască cauzală, KV-cache, decodare token-cu-token. Un diffusion LM cere **atenție bidirecțională**
(fără mască cauzală) și o **buclă de denoising** peste întreaga secvență — lucruri pe care
llama.cpp nu le suportă nativ. Folosirea lui direct, "la pachet", nu e posibilă.

## Decizie
- **Nu** folosim llama.cpp ca atare pentru a rula modelul nostru.
- Construim un **runtime/binar custom** pentru difuzie. **Punctul de pornire** e biblioteca de
  tensori **ggml** (kernel-uri SIMD existente: AVX2 / AVX-512 / AVX-512 VNNI / NEON, cuantizare
  Q4_K/Q8_0, atenție, matmul) pentru a obține rapid **corectitudine + un baseline**.
- **Dar adaptăm kernel-urile la pattern-ul nostru de calcul** (aici e contribuția noastră centrală),
  pentru că llama.cpp/ggml e optimizat pentru **generare autoregresivă**:
  - **Decode AR** = GEMV (1 token, batch mic) + KV-cache + mască cauzală.
  - **Difuzia noastră** = GEMM **batched** (256 tokeni simultan, [256 × hidden] × weights),
    **fără KV-cache**, cu **atenție completă non-cauzală**.
  Acestea au profil diferit de compute/bandwidth, deci kernel-urile stock nu sunt maxime pentru noi.
- Abordare în cascade:
  1. **Baseline:** rulează corect cu kernel-urile ggml stock (referință de corectitudine).
  2. **Adaptare:** kernel de **atenție non-cauzală** (fără mască/cache) + tunning GEMM la
     shape-uri de 256 tokeni; layout-uri/fuziuni potrivite fluxului de denoising.
  3. **Maximizare eficiență:** cai micro-arhitecturale — **AVX-512 VNNI** (VPDPBUSD int8 dot),
     **AMX** (Intel tile ops, int8 MAC), **NEON I8MM/DotProd**, **SVE**, **AMX Apple**;
     kernel-uri proprii unde calea ggml e suboptimă pentru difuzie.

## Alternative respinse
- **llama.cpp direct** — nu suportă difuzie (confirmat prin cercetare); doar cu fork + patch greu.
- **SIMD scris 100% de la zero, fără a porni de la ggml** — inutil de complex; ggml dă un baseline
  corect și kernel-uri deja îngrijite. Pornim de la el și adaptăm.
- **PyTorch/HF** — comod pentru experiment, dar nu target-ul (CPU-only, viteză mare).

## Consecințe
- Scriem **și partea de difuzie** (atenție full non-cauzală + buclă 24 pași + sampler) **și adaptarea
  kernel-urilor** la pattern-ul nostru (GEMM batched, fără cache). Ambele sunt contribuția centrală.
- Kernel-urile adaptate sunt **specifice arhitecturii noastre**, deci parte din codul nostru —
  exact cerința inițială ("codul care modifică modul în care calculează CPU").
- Riscul principal: tunningul kernel-urilor validează și (re)tunează achiziția SIMD per micro-arhitectură;
  măsurăm cu benchmark pe Apple Silicon + VPS AMD/Intel.
- Compatibilitate: ieșirea poate fi GGUF (portabilitate), dar rulează în binarul nostru.
