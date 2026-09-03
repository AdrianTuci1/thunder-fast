# Porting an autoregressive LLM to discrete masked diffusion

This document explains how we turn an autoregressive (AR) LLM — in our case **Qwen2-0.5B** — into a
discrete, non-autoregressive **Masked Diffusion Model (MDM)**. The goal is to keep the pretrained
weights and change only *what the model does*: instead of predicting the next token, it denoises a
sequence starting from a fully `[MASK]` state.

The implementation lives in `src/train/model.py` (`DiffusionLM`), `src/train/diffusion.py`
(`MaskedDiffusion`), `src/train/train.py` and `eval/eval.py`.

---

## 0. What "discrete masked diffusion" means

- **Autoregressive (AR):** generates token by token; at step *t* it sees only the prefix `x[0..t-1]`.
- **Discrete masking (MDM):** starts from a sequence where a fraction of positions are **masked**
  (replaced with the special `[MASK]` token) and reconstructs *all* the masked positions **in
  parallel**, at each diffusion step.

The practical difference: in AR the cost grows with sequence length; in MDM the cost grows with the
**number of denoising steps** (fixed, e.g. 24), independent of window length.

---

## 1. The four structural changes

### 1.1 Bidirectional attention

The AR model uses a **causal mask** (position *i* sees only positions ≤ *i*). For denoising we need
every position to see the **whole context** (including positions to the right), so that it can
reconstruct masked tokens using neighbours on both sides.

**Caution (HF version trap):** setting `is_causal=False` on the modules is **not sufficient**.
The `_update_causal_mask` method in transformers internally builds a 4-D **causal** mask whenever
`attention_mask` is `None` (or 2-D), and that mask is passed directly to the attention kernel and
**overrides** `is_causal`. The correct solution:

```python
L = input_ids.shape[1]
attn_mask = torch.zeros((1, 1, L, L), dtype=torch.bool, device=input_ids.device)  # all-unmasked
out = backbone(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
```

A 4-D all-zero mask (all-`False` for bool) is used directly by `_update_causal_mask`, so all positions
attend to each other. Without this mask the model stays causal and produces degenerate outputs
(it repeats high-frequency tokens, unconstrained by the prompt) both at training and at inference.

> **Mandatory check:** the behaviour of `_update_causal_mask` varies between transformers versions and
> between architectures. If on the chosen architecture the 4-D mask is not used directly, we replace it
> with `_attn_implementation="flash_attention_2"` + a tested all-ones mask.

### 1.2 Discrete mask token `[MASK]`

We add a special `[MASK]` token to the vocabulary (it represents the "noise" / unknown position) and
grow the embeddings:

```python
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
model.resize_token_embeddings(len(tokenizer))
```

At training, the masked positions get `input_ids = mask_token_id`; at inference, the positions to
generate start as `[MASK]` and are revealed progressively.

### 1.3 Re-targeting the head at masked-token reconstruction

Instead of "next-token" cross-entropy, the objective is **reconstruction** of the clean tokens at the
masked positions. `forward(input_ids) -> logits [B, L, V]` produces logits at every position; the loss
is applied **only** on the `[MASK]` positions.

### 1.4 Training objective (masked CE + a "path" term)

```python
mdm_loss = CE(logits[mask], x0[mask])            # reconstruction of masked positions
path_loss = exp(-mdm_loss) * mdm_loss * (1/mask_ratio)  # re-weighting along the denoising path
loss = mdm_loss + path_loss
```

Mask ratios are sampled uniformly from an interval (e.g. `[0.002, 0.998]`), so the model learns to
handle from very few to almost all masked positions.

---

## 2. Generation (inference)

Since the model reconstructs masked positions, generation is **progressive unmasking**:

1. Initialize the positions to generate as `[MASK]`.
2. At each step: `logits = forward(x)`, then a **next-token shift**
   (`logits = cat([logits[:, :1], logits[:, :-1]], dim=1)`) — the same alignment as training.
3. Sample candidate tokens + a **confidence metric** (default **entropy**; alternatives
   `topk_margin`, `p2`).
4. Reveal the most "confident" positions; the rest stay masked. Repeat for the remaining steps.

Unmask algorithms: `entropy` (default), `p2` (re-masking of low-confidence positions),
`origin` (random transfer). The `alg_temp` parameter (default `0.6`) softens the distribution of the
unmask positions.

### Long outputs — block-wise generation

A window is trained on `seq_len` (e.g. 256), but we want up to 2048 tokens. We use **block-wise
generation**: generate a block of `block_len` tokens, append it as a prefix, then generate the next
block. The context grows at each step. `stop_at_eos` stops earlier if an `EOS` appears.

> **Known limitation:** the aggregated context exceeds the training window, so the last blocks can
> degrade (repetition). Solution: longer windows at training, or including long context.

---

## 3. Where it is implemented

| Component | File | Role |
|---|---|---|
| Model + adaptation | `src/train/model.py` | `make_bidirectional`, `[MASK]`, `forward`, `generate`/`generate_long` |
| MDM objective | `src/train/diffusion.py` | `MaskedDiffusion` (masked CE + path loss) |
| Training | `src/train/train.py` | training loop, optimizer, resume |
| Data | `src/train/data.py` | `PackedDataset` |
| Eval | `eval/eval.py` | `reconstruction_loss` + generation |
| Reference inference | `infra/modal_infer_open.py` | `_discrete_generate_window` / `_discrete_generate_long` |
| Runtime (ggml) | `runtime/` | the final target (ADR 0012), ggml kernels over bidirectional attention |

---

## 4. Validation (smoke test)

Before the long run, we validate the whole chain with a small `--max-steps` (e.g. 40–80) on the cloud,
in this order:

1. the model initialises + `forward` works (no crash/OOM),
2. **the loss decreases** (no NaN/inf),
3. save/reload checkpoint (`save_ckpt` → `.pt` + `.meta.json`),
4. upload to R2 + resume from `step_X`,
5. generation at 24 steps (decoding → not garbage/special-only),
6. `reconstruction_loss` is finite,
7. fits in GPU memory (otherwise reduce `batch_size_seq`).

Bidirectional attention is checked explicitly in steps 2/5 (without the 4-D mask the model would stay
causal).
