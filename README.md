# thunder-fast

A non-autoregressive **discrete masked diffusion** model adapted from **Qwen3-0.6B**. It reconstructs a
whole 256-token window in parallel over 24 diffusion steps, using bidirectional attention and a
`[MASK]` token. Long outputs (up to 2048 tokens) are chained block-wise in 256-token windows.

The model is published at **staticlabs/thunder-dlm-0.6b** and runs on Modal/RunPod images, so there is
nothing to install locally.

## Autoregressive vs diffusion

An autoregressive model writes text one token at a time. At step *t* it sees only the prefix `x[0..t-1]`
and predicts the next token, so generating N tokens takes N forward passes and a growing KV-cache, and
it can only continue left to right.

This model instead starts from a sequence of `[MASK]` tokens and reconstructs the whole window at once.
Each of the 24 steps sees the full context (bidirectional attention) and fills in the most confident
positions, so cost scales with the 24 steps rather than the window length, there is no KV-cache, and
because every position attends to every other it can do infilling, not just left-to-right continuation.

## How to use it

```
TF_MODEL=staticlabs/thunder-dlm-0.6b modal run infra/modal_download_open.py
TF_PROMPT="Write a quick sort algorithm in python." modal run infra/modal_infer_open.py
```

`TF_MODE=single` uses one large window instead of the default block-wise mode. Training is
`modal run infra/modal_train.py -- --config config/train_config.yaml`, and evaluation is
`python eval/eval.py --ckpt <path>`. The technical details of the adaptation are in
[`docs/porting-to-diffusion.md`](docs/porting-to-diffusion.md).

## Speed

On an A10G (bf16, same 0.5B checkpoint) the reference PyTorch engine generates a 256-token window in
about 1.5 s at 24 steps, i.e. roughly **180 tok/s**; the 512-token block-wise mode runs at about
**215 tok/s**. For comparison, the same weights decoded autoregressively (greedy, causal attention)
reach about **43 tok/s** — but that harness has no KV-cache and recomputes the whole sequence at each
step, so a production AR decoder with KV-cache is substantially faster per stream.
