# Infrastructure

Training and inference run on cloud images; there is nothing to install locally. Credentials
(R2, HF, WandB, RunPod) are **environment variables / Modal secrets only** — never committed.

## Modal

- `infra/modal_train.py` — training runner. Spawns a long (24h) training run on an H100 GPU.
  The repo source is copied into the container at `/repo`; the config is read as a string on
  the client and parsed inside the container. Checkpoints go to the persistent
  `thunder-checkpoints` Volume at `/vol/checkpoints`. `.spawn()` fires and returns immediately
  so the client can disconnect; `.remote()` blocks and streams logs (for a smoke test).
  ```
  TF_CONFIG=config/train_config.yaml TF_MAX_STEPS=60 TF_CKPT_EVERY_STEPS=20 modal run infra/modal_train.py
  ```
  Secrets: `hf` (HF_TOKEN), `wandb` (WANDB_API_KEY). GPU via `TF_GPU` (default H100).
- `infra/modal_infer_open.py` — long-output (up to 2048) inference via the block-wise engine;
  `TF_MODE=single` selects one large window. `TF_MODEL_DIR` points at the model on the volume.
- `infra/modal_download_open.py` — downloads the base model into the volume.
- `infra/modal_r2_transfer.py` — CPU-only job that picks the newest `step_*.pt` on the volume
  and pushes it (plus its `.meta.json`) to R2. Resumes rely on `train.py` reading the R2 prefix.
  ```
  modal run infra/modal_r2_transfer.py [-- --step 7591]
  ```

## RunPod

- `infra/Dockerfile.runpod` — the training container. Build + push:
  ```
  docker build -f infra/Dockerfile.runpod -t <registry>/thunder-fast-train:latest .
  docker push <registry>/thunder-fast-train:latest
  ```
  The default entrypoint runs `src/train/train.py --config config/train_config.yaml`.
- `infra/runpod_launch.py` — convenience wrapper over the RunPod jobs API. It passes the R2
  env vars and runs training; add `--upload-r2` to also push checkpoints to R2:
  ```
  RUNPOD_API_KEY=... R2_ENDPOINT=... R2_BUCKET=... R2_ACCESS_KEY=... R2_SECRET_KEY=... \
    python infra/runpod_launch.py --image <image> --gpu A100 --upload-r2
  ```

## R2 (Cloudflare object storage)

`infra/r2.py` is a shared boto3 client (S3-compatible), usable on Modal, RunPod, or a local
machine. It accepts either the canonical env names (`R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY,
R2_SECRET_KEY`) or the Modal/AWS-style names (`R2_ENDPOINT_URL, R2_BUCKET, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY`).

Functions: `client()`, `upload_file`, `download_file`, `exists`, `list_keys`, `upload_dir`,
`download_dir`. Checkpoints are stored under `storage.r2_prefix` (default
`thunder-fast/checkpoints`), matching the Modal transfer entrypoint and `train.py`'s resume.

The dispatch workflow (`build-runtime.yml`) reads R2 creds from GitHub secrets and uploads the
runtime build with `runtime/tools/upload_r2.py`.
