# Adapting a 30B MoE to discrete masked diffusion (encoder–decoder)

This document shows, as diagrams, **what an MoE diffusion model looks like** and **what we add** to turn a
causal autoregressive 30B MoE into a discrete masked-diffusion model. It is written for the base
**`Qwen/Qwen3-30B-A3B`** we will adapt.

We do **not** adopt a ready-made diffusion model here — we port our own MoE to diffusion.

**What the base already gives us (verified `config.json`):**

| Property | Value |
|---|---|
| Total / active params | 30.5B / 3.3B |
| Layers | 48 |
| Attention | full **causal**, **no sliding window**; 40K native → YaRN ~128K |
| MoE | **128 experts / 8 active**, `norm_topk_prob`, `router_aux_loss_coef=0.001` |
| Hidden / heads | 2048 / 32 Q · 4 KV, `head_dim=128` |
| FFN | dense `intermediate=6144`, MoE `moe_intermediate=768` |
| Vocab | 151936 |
| License | Apache 2.0 |
| Form | decoder-only AR, `Qwen3MoeForCausalLM` |

**What we add** (the diffusion port): an **encoder + decoder split**, a **cross-attention** decoder over a
**256-token canvas**, a **`[MASK]`** token + **discrete masked cross-entropy** objective, and a
**progressive un-mask sampler**. A **sliding-window** attention is *not* required for ≤128K (see §3);
it is an optional later optimization to push encoder prefill beyond that.

---

## 1. One MoE transformer block — what we already have

An "expert" is **not** a whole model. In every block there is **one** shared attention network and
**many** expert FFNs. Every token flows through the same shared attention; a router then picks which
experts (FFNs) process it. Qwen3-30B-A3B ships this block natively.

```
                hidden state h  [tokens, hidden]
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  SHARED — attention (ONE copy, used by ALL experts)          │
   │                                                              │
   │   ┌──────────────────────────────────────────────────────┐   │
   │   │  Q/K/V projection                                     │   │
   │   │  + attention                                        │   │
   │   │     causal full     (encoder, autoregressive)  OR     │   │
   │   │     sliding-window  (encoder, optional long ctx) OR   │   │
   │   │     bidirectional   (decoder canvas)           OR     │   │
   │   │     cross-attention (decoder -> ctx cache)            │   │
   │   │  + o_proj + residual + norm                           │   │
   │   └──────────────────────────────────────────────────────┘   │
   │                              │  h'  (identical for every expert)│
   │                              ▼                                │
   │  ┌────────────────────────────────────────────────────────┐  │
   │  │ MoE FFN — the ONLY per-expert part (routed)           │  │
   │  │                                                      │  │
   │  │  gate(h')  ──►  softmax  ──►  top-8 of 128 + shared   │  │
   │  │                                                      │  │
   │  │  out = Σ_j w_j · FFN_expert_j(h')                    │  │
   │  │       (w_j = gate weight; FFN_expert_j = expert MLP) │  │
   │  └────────────────────────────────────────────────────────┘  │
   │                              │  + residual + norm           │
   │                              ▼                              │
   └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                hidden state h''  [tokens, hidden]
```

```
LEGEND
  [SHARED]   attention + projections + norms = trained once, reused by every expert
  [MoE]      the router + the N expert FFNs  = the only thing that is per-expert
```

Key takeaway: **only the FFN weights are multiplied.** Attention, embeddings, norms stay **shared**, so a
bigger total model costs only the *active* experts per token. This is already true of the 30B base.

---

## 2. Encoder–decoder diffusion structure (what we build on top)

The base is decoder-only. For a diffusion model that can denoise a whole canvas while keeping *very long*
context cheap, we add a **causal encoder** that caches the prompt once, and a **bidirectional decoder** that
denoises a 256-token canvas and reads the cache via **cross-attention**.

```
   PROMPT / CONTEXT  (up to ~128K, YaRN)           CANVAS  (256 tokens to denoise)
 ┌────────────────────────────────────────┐         ┌─────────────────────────────────────┐
 │            ENCODER (autoregressive)    │         │          DECODER                    │
 │                                        │         │                                     │
 │  h -> [block] -> [block] -> ... -> ctx │         │   [MASK][MASK]...[MASK]  (256)      │
 │                                        │         │               │                     │
 │  per block (reuses §1 block):          │         │   per block (reuses §1 block):      │
 │    self-attention (causal, optional    │         │     self-attention                  │
 │        sliding-window) + KV cache      │         │        (bidirectional, over canvas) │
 │    -> MoE FFN (8/128 experts)          │         │     cross-attention ──► encoder KV  │
 │                                        │         │        (Q=canvas, K/V=encoder)      │
 └───────────────────┬────────────────────┘         │     -> MoE FFN (8/128 experts)      │
                     │                              │     -> output head -> logits        │
                     │  encoder KV cache            │               │                     │
                     │  (memory, cross-attn target) │               ▼                     │
                     └──────────────►───────────────►      un-mask lowest-entropy tokens   │
                                                              repeat up to ~48 denoise steps│
                                                              └─────────────────────────────┘
```

```
FLOW
 1. ENCODER runs autoregressively over the prompt and builds a KV cache. Cost is linear in
    prompt length — later layers do NOT re-read the whole prompt.
 2. DECODER starts from a 256-token canvas filled with [MASK] and denoises all 256 positions in
    parallel with FULL (bidirectional) attention.
 3. It reads prompt history via CROSS-ATTENTION to the encoder cache — not by re-encoding.
 4. Multi-canvas: when a canvas is clean, append it to the context, re-run the encoder, and
    denoise the next 256-token canvas.
```

Why the encoder/decoder split even though the base has no SSM: it makes **long-context cheap**. In a
decoder-only bidirectional model every denoise step re-attends to the *whole* prompt (O(N²) per step),
so a 128K prompt costs ~O(N²) × 24–48 steps. Encoding the prompt *once* and reading it through
cross-attention makes each step O(canvas² + canvas·context) — the win DiffusionGemma exploits.

---

## 3. What we ADD to port the 30B MoE to diffusion

The MoE part is already there; the diffusion part is what we add.

| Autoregressive (base) | What we change / add | Why |
|---|---|---|
| causal self-attention | add **bidirectional** attention on the decoder canvas | a masked position must see both neighbours |
| recompute on each token | add an autoregressive **encoder + KV cache**, and a decoder **cross-attention** to it | cheap long context (encode once, reuse) |
| one shared FFN (routed MoE) | keep it — already the "shared attention + experts" design | big total model, small active compute |
| next-token cross-entropy | re-target the head at **discrete masked cross-entropy** + `[MASK]` token | denoise masked positions, not predict next |
| greedy / left-to-right decode | add a **progressive un-mask sampler** (entropy-bounded, up to ~48 steps, adaptive stop) | iterative parallel denoising |
| decoder-only | add the **encoder–decoder split** | denoise a canvas while keeping long context cheap |

**About sliding window** (your "fereastra glisantă"): Qwen3-30B-A3B has **none**. We do **not** need it for
the cache win — the encoder-decoder split already avoids re-reading the prompt per step. We only add
sliding-window attention (a local-attention modification, needs re-training/tuning) if we want encoder
prefill cheap beyond ~128K. For ≤128K, the full-attention encoder + YaRN is enough.

### 3.1 Shared attention + expert specialization

- The encoder/decoder attention is **one shared network**; it works for *all* experts because each expert
  gets the same `h'`.
- To specialize for a task we **freeze** the shared attention and only tune:
  - the **expert FFN** for that task, and
  - the small **router** (so the task tokens are routed to that expert).
- Cost = expert FFN + router, not the whole active model. That is the cheap "one big model, train a slice" path.

---

## 4. Q4 fit on 24GB & where this fits in the repo

- **30.5B × ~0.58 B/param (Q4_K_M) ≈ 17.7GB**; Q4_0 ≈ 17.1GB; IQ4_XS ≈ 16.3GB. Active 3.3B → fast forward
  passes. Fits a 24GB GPU (RTX 3090/4090, A10G) with room for KV + activations; only ~8 experts + shared
  are active at a time (inactive experts can be offloaded).
- This is the architecture we build for the ~30B MoE target. `docs/porting-to-diffusion.md` stays the
  description of the simpler 0.6B decoder-only model. The 0.6B pipeline and runtime remain untouched;
  the encoder–decoder + MoE path is additive.
