# Inference

There are three ways to run the model. This page covers the reference PyTorch path and the
speeds; see [runtime](runtime.md) for the standalone ggml engine.

## Reference PyTorch path (src/infer/inference.py)

Loads a trained checkpoint, runs the reverse-diffusion loop for a fixed output length, decodes,
and reports throughput:

```
python src/infer/inference.py --ckpt <path.pt> --prompt "Bună ziua, eu sunt" [--gpu]
```

- `--base-model` overrides the checkpoint's recorded base model id.
- `--gpu` uses CUDA; the default is CPU.
- It runs `model.generate(...)` once for the window length (`--target`), then reports
  `tokens/sec` and the decoded text.

The same diffusion core powers `eval.py` (see [evaluation](evaluation.md)).

## Block-wise generation

`DiffusionLM.generate_long(...)` chains windows: one diffusion pass produces `block_len` tokens
(256 by default), appends them to the growing context, and repeats up to `max_new_tokens`
(2048). This is the default long-output engine used for the modal inference entrypoint.

## Measured speed (A10G, bf16, ~0.5B, 24 steps)

| Mode | Tokens/sec | Note |
|---|---|---|
| Single window (256) | ~180 | ~1.5 s for ~270 tokens |
| Block-wise (512) | ~215 | chained windows |
| Autoregressive (same weights, greedy) | ~43 | lower bound — this harness has no KV-cache and recomputes the whole sequence each step |

Diffusion is faster here because it does 24 full-window Batched GEMM passes instead of N
per-token GEMV passes, and attention is fully computed once (no cache).

## Vector/format notes

The reference engine operates on the `.pt` checkpoint (state_dict) and the model's own
tokenizer. For CPU/Apple/AMD local deployment there is a separate GGUF runtime — see
[runtime](runtime.md).
