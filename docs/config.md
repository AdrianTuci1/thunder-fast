# Config reference (`config/train_config.yaml`)

All values are read by `src/train/train.py` (and `eval/eval.py` for the eval section). Nothing
is hardcoded, so the same pipeline scales to a different backbone / budget by editing this file.

## model

| Key | Default | Meaning |
|---|---|---|
| `base` | `Qwen/Qwen3-0.6B` | The Hugging Face causal backbone id. Change this to scale the model size (e.g. a 7B); the training/inference code is unchanged. |
| `use_gqa` | `true` | Grouped-query attention on the backbone. |

## diffusion

| Key | Default | Meaning |
|---|---|---|
| `seq_len` | `256` | Generated window length (also the packed sequence length). |
| `infer_steps` | `24` | Un-mask steps at inference. Training has no diffusion steps; this only drives progressive un-masking. |
| `mask_ratio_min` | `0.002` | Lower bound of the per-sample uniform mask-ratio curriculum (`1/500`). |
| `mask_ratio_max` | `0.998` | Upper bound (`1 - 1/500`). Min>0 and max<1 so sequences are never fully clean or fully masked. |

## training

| Key | Default | Meaning |
|---|---|---|
| `total_tokens` | `2_000_000_000` | Token budget; used to estimate total steps for the cosine schedule. |
| `batch_size_seq` | `32` | Sequences per micro-batch (each `seq_len` long). |
| `grad_accum` | `8` | Micro-batches per optimizer step. Global batch = `batch_size_seq * seq_len * grad_accum`. |
| `num_workers` | `0` | DataLoader workers. `0` is the most robust against a known streaming deadlock; streaming is I/O-bound anyway. |
| `gradient_checkpointing` | `true` | Trade compute for memory. |
| `learning_rate` | `2.0e-4` | Base LR (conservative, to preserve the AR initialization). |
| `lr_schedule` | `cosine` | Cosine decay toward ~10% of peak. |
| `warmup_steps` | `2000` | Linear warmup length. |
| `weight_decay` | `0.01` | AdamW weight decay. |
| `max_grad_norm` | `1.0` | Gradient clipping. |
| `bf16` | `true` | bf16 training. |
| `max_time_hours` | `4.0` | Graceful wall-clock stop (0 = none, use `max_steps`). |
| `ckpt_every_hours` | `2.0` | Time-based checkpoint interval. |
| `keep_last` | `3` | Keep only the N most recent checkpoints (rotation). |

## data

| Key | Default | Meaning |
|---|---|---|
| `sources` | c4:en 0.60, c4:ro 0.30, opus-100:en-ro 0.10 | Weighted mix. Each source has `name` (`repo[:config]`), `type` (`mono`/`parallel`), `weight`. |
| `packing` | `true` | Pack sequences to `seq_len`. |
| `bos_ratio` | `0.045` | Fraction of documents prefixed with BOS to learn natural starts. |

## eval

| Key | Default | Meaning |
|---|---|---|
| `datasets` | `liro`, `flores200-en-ro`, `wmt19-en-ro` | Romanian NLU / MT benchmarks. |
| `run_every_steps` | `1000` | How often to run eval from the training loop. |
| `max_samples` | `512` | Cap for eval samples. |

## storage

| Key | Default | Meaning |
|---|---|---|
| `r2_prefix` | `thunder-fast/checkpoints` | Object-store prefix used by `train.py --upload-r2`, `_find_latest`, and `modal_r2_transfer.py`. |
