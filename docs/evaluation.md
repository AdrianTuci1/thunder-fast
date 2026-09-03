# Evaluation

The harness is `eval/eval.py` — deliberately lightweight for a PoC:

```
python eval/eval.py --config config/train_config.yaml --ckpt <path.pt> [--prompt ...] [--steps N]
```

It measures and prints:

- **Reconstruction (denoising) loss** on held-out text for the first configured source. A
  down-trending value means the model is learning (not NaN/inf). The loss is the same
  `MaskedDiffusion.training_loss` used during training, averaged over `limit` samples.
- **A sample()** decoded to text for a quick qualitative look. This is not expected to be
  coherent early on — the sanity check is that it isn't garbage/special-token-only.

The device is CUDA if available, otherwise CPU.

## Wired-but-not-implemented benchmarks

The full Romanian NLU (LiRo) / MT (FLORES, WMT) benchmarks are stubbed so the harness can be
extended once a good checkpoint exists (`config.eval.datasets`: `liro`, `flores200-en-ro`,
`wmt19-en-ro`). `config.eval.run_every_steps` and `max_samples` are already in the config for
calling evaluation from the training loop.

There is also `infra/diag_loss.py` (a quick loss diagnostic) and `infra/bench_gpu.py` (GPU
throughput benchmark) in the infra directory.
