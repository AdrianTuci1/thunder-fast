"""Modal CPU-only runner: push the latest training checkpoint (volume -> R2).

Usage (from the repo root):
    modal run infra/modal_r2_transfer.py
    modal run infra/modal_r2_transfer.py -- --step 7591

The runner mounts the `thunder-checkpoints` Volume, picks the newest `step_*.pt` on it
(unless `--step` is given), and uploads it plus its `.meta.json` to R2 under the same
`storage.r2_prefix` the trainer uses (`thunder-fast/checkpoints`). That way a RunPod job
with R2 credentials in env can resume via `train.py`'s `_find_latest(r2_prefix)` path.

It is intentionally CPU-only (no GPU): transferring a checkpoint only needs boto3 + the repo
`infra/r2.py`, so a slim Python image is enough.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = "/vol/checkpoints"
REPO_REMOTE = "/repo"
R2_PREFIX = "thunder-fast/checkpoints"  # must match config `storage.r2_prefix`


def _ignore(p: Path) -> bool:
    """Return True for files/dirs we don't want baked into the container image."""
    name = p.name
    if name.startswith("."):  # .env, .git, .agents, ...
        return True
    if name in {"checkpoints", "__pycache__", "node_modules", ".venv", "venv"}:
        return True
    if name.endswith((".pyc", ".pt", ".log", ".png", ".pdf", ".sqlite", ".sqlite.bak", ".bin")):
        return True
    if p.suffix in {".yaml", ".yml"}:
        return True
    return False


app = modal.App("thunder-fast-r2-transfer")

# CPU-only: boto3 is the only dep needed to read the volume and write to R2.
image = (
    modal.Image.from_registry("python:3.11-slim")
    .pip_install("boto3>=1.34")
    .add_local_dir(REPO, remote_path=REPO_REMOTE, copy=True, ignore=_ignore)
)

# Reuse the existing R2 secret (see infra/r2.py for the exact env-var names it reads).
r2_secret = modal.Secret.from_name("r2-credentials")
checkpoints_volume = modal.Volume.from_name("thunder-checkpoints", create_if_missing=True)


def _latest_on_volume(d: Path) -> int | None:
    steps = []
    for f in d.glob("step_*.pt"):
        m = re.match(r"step_(\d+)\.pt", f.name)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


@app.function(
    image=image,
    cpu=4.0,
    memory=4096,
    secrets=[r2_secret],
    volumes={CHECKPOINT_DIR: checkpoints_volume},
    timeout=3600,
)
def transfer_step(step: int | None = None) -> str:
    sys.path.insert(0, REPO_REMOTE)

    # Print which R2_* env keys are present (names only, never values) so a naming
    # mismatch with infra/r2.py is easy to spot before the upload.
    keys = sorted(k for k in os.environ if k.startswith("R2_"))
    print(f"[r2-transfer] R2_* env keys present: {keys}", flush=True)

    from infra.r2 import exists, upload_file

    if step is None:
        step = _latest_on_volume(Path(CHECKPOINT_DIR))
    if step is None:
        raise FileNotFoundError("no step_*.pt found on the volume; pass --step")

    ckpt = Path(CHECKPOINT_DIR) / f"step_{step}.pt"
    meta = Path(CHECKPOINT_DIR) / f"step_{step}.meta.json"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not on the volume")

    print(f"[r2-transfer] uploading step_{step} ({ckpt.stat().st_size / 1e9:.2f} GB) -> {R2_PREFIX}/", flush=True)
    for local, suffix in [(ckpt, "pt"), (meta, "meta.json")]:
        key = f"{R2_PREFIX}/step_{step}.{suffix}"
        upload_file(local, key)
        print(f"[r2-transfer] uploaded {local.name} -> {key} (exists={exists(key)})", flush=True)
    return f"uploaded step_{step}"


@app.local_entrypoint()
def main(step: int | None = None) -> None:
    result = transfer_step.remote(step)
    print(result, flush=True)
