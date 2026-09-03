# Training

The training loop is in `src/train/train.py`; the entrypoint is `run_train(cfg, out, ...)`.
The job runs on Modal or RunPod, checkpoints periodically, and (optionally) uploads each
checkpoint to Cloudflare R2 so the run is resumable.

## Objectives / key parameters

Read from `config/train_config.yaml` (see [config](config.md)):

- `model.base` — the HF backbone id (e.g. `Qwen/Qwen3-0.6B`). Changing this is how you scale
  to a larger model; the loss and checkpointing code are unchanged.
- `diffusion.seq_len` (256) — generated window length.
- `diffusion.infer_steps` (24) — un-mask steps at inference. Training has *no* diffusion steps;
  the step count appears only during progressive un-masking.
- `diffusion.mask_ratio_min/max` (0.002 / 0.998) — mask-ratio curriculum bounds.
- `training.*` — budget (`total_tokens`), batch (`batch_size_seq * seq_len * grad_accum`),
  optimizer/LR schedule, gradient-checkpointing, time-based checkpointing
  (`ckpt_every_hours`, `keep_last`).
- `data.sources` — weighted mix of HF datasets (mono + parallel).

## Data pipeline (src/train/data.py)

`PackedDataset` streams a weighted mix of Hugging Face datasets and packs token sequences to
`seq_len`:

- draws a source proportional to its weight (and cycles so a small source is oversampled
  rather than stalling the stream),
- handles mono (`text`) and parallel (`translation`) rows —
  for parallel sources it joins the source/target language text so the model sees both,
- tokenizes with the diffusion model's tokenizer,
- optionally prefixes documents with `BOS` (`bos_ratio`, to learn natural text starts),
- packs tokens into fixed-length sequences.

Unavailable/gated sources are skipped (added to a dead set) instead of killing training.

## Training loop (src/train/train.py)

- **One AdamW group** over the whole model; we continue pretraining the entire AR model on the
  masked-reconstruction objective, so the single new `[MASK]` embedding row learns with the
  rest at the base LR.
- **Cosine LR schedule** over an estimated step count from the token budget.
- Each optimizer step accumulates `grad_accum` micro-batches, clips gradients, steps, and
  zeroes.
- **Resume:** if a checkpoint exists in `--out` (or in R2 when `--upload-r2`), it restarts from
  that step. If a checkpoint is structurally incompatible with the current objective (the
  discrete rewrite changed the state dict), it starts clean from step 0 with a fresh optimizer.
- **Checkpointing:** periodically by wall-clock (`training.ckpt_every_hours`) or by step
  (`--ckpt-every-steps`, for a smoke test), keeping only the last `keep_last`. A
  `SIGTERM` handler saves a final checkpoint so a cloud stop resumes from the current step.

## Checkpoints

`torch.save` writes `step_<N>.pt` (model + optimizer + scheduler + step + tokens_seen +
`base_model`) plus `step_<N>.meta.json` (step, tokens_seen, base_model, created_at).

## R2 (optional)

`train.py --upload-r2` pushes each checkpoint pair to R2 under `storage.r2_prefix` and resumes
from R2 if no local checkpoint exists. The credentials come from the environment
(`R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY, R2_SECRET_KEY`) via `infra/r2.py`. This is
platform-agnostic — the same flag works on RunPod and Modal.

See [infrastructure](infrastructure.md) for how training is launched in each cloud.
