# Conversion

There are two conversion steps, for two different targets.

## AR → diffusion checkpoint (convert/convert_to_diffusion.py)

Produces the **initial** discrete MDM weights from a pretrained autoregressive LLM: the base
AR weights plus a newly added `[MASK]` token row in the embedding matrix and LM head, so
training has a clean start. It writes:

```
model/config.json                       bidirectional flag, mask_token_id, hidden_size, vocab_size
model/diffusion_model.safetensors       the full adapted weights
```

```
python convert/convert_to_diffusion.py --base-model Qwen/Qwen3-0.6B --out model
```

This is the training-side conversion.

## Checkpoint / safetensors → GGUF (runtime/tools/convert_to_gguf.py)

Produces the GGUF the ggml runtime loads. It reads `config.json` + `diffusion_model.safetensors`
and writes a GGUF with the hyperparams under `qwen3.*` and the diffusion params under `dlm.*`
(matching `runtime/src/engine/model.h`). Weight keys map the HF/DiffusionLM names to ggml names
(`token_embd.weight`, `blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`, `output.weight`, ...).

```
python runtime/tools/convert_to_gguf.py --checkpoint model --out model.gguf
```

The converter also embeds the backbone's **tokenizer** into the GGUF, so the runtime loads it
from the model file and does not require a separate directory. It runs in the CI/Modal build
environment (needs `gguf` + `safetensors`).

> Note: the tensor-name mapping and the tokenizer embedding are validated against the exact
> checkpoint and ggml version at build time; local dev has neither dependency.
