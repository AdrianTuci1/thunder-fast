# Architecture

## The idea

An autoregressive (causal) transformer predicts the *next* token from a growing prefix. A
masked-diffusion model instead reconstructs a whole window from a partially/noisy view. The
training objective is masked token reconstruction (cross-entropy only over masked positions),
and generation is an iterative un-masking loop over the full window.

Adapting a causal checkpoint to diffusion needs three structural changes:

1. **Causal → bidirectional attention.** Every position must attend to every position.
2. **A discrete `[MASK]` token** added to the vocabulary (the "noise" token).
3. **Re-target the output head** at masked-token reconstruction rather than next-token
   prediction.

We keep the original weights and tokenizer and wrap the base transformer, so the model is
indifferent to the base architecture's size.

## `DiffusionLM` (src/train/model.py)

`DiffusionLM(base_model_id)` wraps **any** Hugging Face causal LM via
`AutoConfig` / `AutoModelForCausalLM` / `AutoTokenizer`. None of the dimension constants are
hardcoded — `hidden_size`, `vocab_size`, layer count, etc. are read from the config. This is
what makes the same code reusable for a larger backbone (e.g. a 7B) with a config change, not
a rewrite.

Construction:
- adds a `[MASK]` token if the base tokenizer lacks one,
- sets `config._attn_implementation = "sdpa"` and `use_cache = False`,
- calls `make_bidirectional(...)`,
- resizes the embedding matrix to include the new `[MASK]` row (grows both the input embedding
  and the LM head).

## Bidirectional attention (critical)

Setting `module.is_causal = False` alone is **not enough**. Hugging Face's `_update_causal_mask`
builds a *causal* 4-D mask whenever `attention_mask` is `None` (or 2-D), and that mask overrides
the module's `is_causal`, silently leaving the model causal. The fix is to pass an explicit
4-D **all-zeros** bool mask at forward time (see `DiffusionLM.forward`), which the SDPA path
uses directly and makes every position attend to every other — MDM denoising mode.

Because the sequence length is 256, a full `O(n²)` attention is cheap (256×256 = 65,536) and
there is **no KV-cache** in this model.

## `MaskedDiffusion` (src/train/diffusion.py)

The discrete MDM objective:
- a per-sample mask ratio is sampled uniformly from `[mask_ratio_min, mask_ratio_max]`,
- those positions are replaced with `[MASK]`,
- the model runs bidirectionally and predicts logits,
- loss = masked cross-entropy (`mdm_loss`) + a confidence-scaled "path" term (`path_loss`),
  computed only over masked positions.

There is no continuous noise schedule and no learned time embedding: the model simply predicts
logits at every position, and the diffusion "time" appears only at inference (how many un-mask
steps have run). The `MaskedDiffusion` object is the single source of truth for the masking
schedule and the inference step count.

## Generation (src/train/model.py)

- `generate(...)` — one fixed window: prompt tokens are held fixed (clean/observed), the
  remaining positions start from `[MASK]` and are revealed most-confident-first over `steps`
  passes. `alg` selects the confidence metric (`entropy` default, `topk_margin`, or the
  `origin`/`p2` variants in the reference).
- `generate_long(...)` — chained block generation for long outputs (up to +2048): generates
  `block_len` tokens per pass, appends them to the growing context, and repeats.

Both call a generic `forward(input_ids) -> logits[B,L,V]`, so the generation loop is independent
of the backbone.
