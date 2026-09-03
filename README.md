# thunder-fast

A non-autoregressive **discrete masked diffusion** model adapted from **Qwen2-0.5B**. It reconstructs a
whole 256-token window in parallel over 24 diffusion steps, using bidirectional attention and a
`[MASK]` token. Long outputs (up to 2048 tokens) are chained block-wise in 256-token windows.

The model is published at **staticlabs/dlm-code0.6b-exp** and runs on Modal/RunPod images, so there is
nothing to install locally.

Complete documentation, including architecture, training, the ggml runtime, and the
infrastructure, is in [`docs/index.md`](docs/index.md).

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
TF_MODEL=staticlabs/dlm-code0.6b-exp modal run infra/modal_download_open.py
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

## Related work

This project adapts a pretrained **autoregressive** model (Qwen2-0.5B) into a masked-diffusion model
rather than training a diffusion LM from scratch. This mirrors the finding behind **DiffuLLaMA**:
training a diffusion language model from scratch at scale remains challenging, whereas starting from a
pretrained AR backbone lets the model learn language first and then be adapted to generate through
diffusion. In our experiments a model trained purely on the diffusion objective from pretraining did
not converge to coherent output, while adapting a pretrained AR backbone is what made the diffusion
generator work.

- [DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA) — *Scaling Diffusion Language Models via
  Adaptation from Autoregressive Models* (ICLR 2025). Converts GPT-2/LLaMA checkpoints into diffusion
  models by continual pre-training, using far less compute than training a diffusion LM from scratch.
- [Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM) — *Diffusion-LM Improves Controllable
  Text Generation* (NeurIPS 2022). The original non-autoregressive diffusion text model, with
  conditional/classifier-guided generation.
- [LLaDA](https://github.com/ML-GSAI/LLaDA) — *Large Language Diffusion with mAsking*. Discrete masked
  diffusion LMs trained at scale, with a masked-diffusion pretraining and decoding schedule related to
  the one used here. (Note: LLaDA is trained from scratch rather than adapted from an AR model.)
