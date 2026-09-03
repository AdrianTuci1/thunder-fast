"""Modal runner for thunder-fast diffusion adaptation training.

Usage (from the repo root):
    TF_CONFIG=config/train_config.yaml TF_MAX_STEPS=60 TF_CKPT_EVERY_STEPS=20 modal run infra/modal_train.py

The repo source is copied into the container at /repo via `Image.add_local_dir` (this is the
supported way in modal 1.5.1; `@app.function` no longer takes a `mounts` kwarg and
`include_source` only statically follows module-level imports, so deferred `src.*` imports
were never uploaded). The config file is read as a raw string on the local machine and parsed
to a `dict` inside the container (which has PyYAML + torch), so no heavy dependency is needed
locally. The CUDA image is built from `requirements.txt`, the HF token is injected as a secret
(needed to download Qwen3-0.6B and gated datasets like OSCAR), and checkpoints are written into
the persistent `thunder-checkpoints` Volume mounted at /vol/checkpoints (only the last N are
kept - see config `training.keep_last`).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = "/vol/checkpoints"
REPO_REMOTE = "/repo"


def _ignore(p: Path) -> bool:
    """Return True for files/dirs we don't want baked into the container image."""
    name = p.name
    if name.startswith("."):  # .env, .git, .agents, ...
        return True
    if name in {"checkpoints", "__pycache__", "node_modules", ".venv", "venv"}:
        return True
    if name.endswith((".pyc", ".pt", ".log", ".png", ".pdf", ".sqlite", ".sqlite.bak", ".bin")):
        return True
    if p.suffix in {".yaml", ".yml"}:  # config is passed as a string, not read from the image
        return True
    return False


app = modal.App("thunder-fast-train")

# torch >= 2.5 is required by this transformers line (Qwen3 support), so the base image must
# ship a matching torch instead of the older 2.4.0 image.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .apt_install("git")
    .add_local_dir(REPO, remote_path=REPO_REMOTE, copy=True, ignore=_ignore)
)

# HF token for downloading the base model + gated datasets. Create it once with:
#   modal secret create hf HF_TOKEN=<token>
hf_secret = modal.Secret.from_name("hf")

# Weights & Biases key for live metric logging. Create it once with:
#   modal secret create wandb WANDB_API_KEY=<key>
wandb_secret = modal.Secret.from_name("wandb")

# Persistent volume for checkpoints. This is Modal's object storage for large files.
# `create_if_missing=True` so a deleted/recreated volume springs back on the next run;
# the trainer also rotates to keep only the last few checkpoints (config `training.keep_last`).
checkpoints_volume = modal.Volume.from_name("thunder-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    secrets=[hf_secret, wandb_secret],
    volumes={CHECKPOINT_DIR: checkpoints_volume},
    timeout=86400,  # 24h headroom
    memory=32768,
)
def train(config_text: str, max_steps: int | None = None, ckpt_every_steps: int | None = None, out_dir: str | None = None):
    import sys

    import yaml

    sys.path.insert(0, REPO_REMOTE)

    from src.train.train import run_train

    cfg = yaml.safe_load(config_text)
    out_dir = out_dir or os.environ.get("TF_OUT") or CHECKPOINT_DIR
    run_train(
        cfg,
        out_dir,
        max_steps=max_steps,
        ckpt_every_steps=ckpt_every_steps,
    )
    return "training finished"


@app.local_entrypoint()
def run():
    # Config is passed via env vars (the `modal run` CLI passes no positional args).
    config = os.environ.get("TF_CONFIG", "config/train_config.yaml")
    max_steps = int(os.environ["TF_MAX_STEPS"]) if os.environ.get("TF_MAX_STEPS") else None
    ckpt_every_steps = int(os.environ["TF_CKPT_EVERY_STEPS"]) if os.environ.get("TF_CKPT_EVERY_STEPS") else None
    gpu = os.environ.get("TF_GPU", "H100")
    out_dir = os.environ.get("TF_OUT")
    block = bool(os.environ.get("TF_BLOCK"))

    # Read the YAML as a raw string locally (no torch/yaml needed on the laptop); the
    # container parses it where PyYAML is installed.
    config_text = (REPO / config).read_text(encoding="utf-8")
    # `.spawn()` (not `.remote()`) fires the long training on Modal's infra and returns
    # immediately, so an ephemeral local client can exit and the 4h run keeps going
    # server-side. The run is independent of this process's lifetime. `.remote()` blocks
    # and streams logs (handy for a short smoke test).
    if block:
        train.remote(config_text, max_steps=max_steps, ckpt_every_steps=ckpt_every_steps, out_dir=out_dir)
    else:
        call = train.with_options(gpu=gpu).spawn(config_text, max_steps=max_steps, ckpt_every_steps=ckpt_every_steps, out_dir=out_dir)
        print(f"[launch] spawned training call {call.object_id}; safe to disconnect", flush=True)
