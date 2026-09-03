"""Modal runner for thunder-fast diffusion inference.

Loads a trained diffusion checkpoint from the persistent `thunder-checkpoints` volume
(or R2) and runs the reference PyTorch reverse-diffusion loop (`model.sample`), reporting
decoded output and throughput (tokens/sec). Runs on CPU (`--gpu ""`) or a small GPU
(`--gpu A10G` / `--gpu T4`).

The diffusion hyper-parameters are read from `config/train_config.yaml` on the local
machine (the same config that trained the checkpoint) and passed as a string, so the
infer steps / mask-ratio curriculum match training exactly.

Usage (from the repo root):
    TF_CKPT=step_14939 TF_GPU=A10G TF_PROMPT="Bună ziua" modal run infra/modal_infer.py
    TF_CKPT=step_14939 TF_GPU=""     modal run infra/modal_infer.py   # CPU
    # profile throughput at seq_len=256:
    TF_CKPT=step_14939 TF_PROFILE_LEN=256 ... modal run infra/modal_infer.py
"""

from __future__ import annotations

import os
import time
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


app = modal.App("thunder-fast-infer")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .apt_install("git")
    .add_local_dir(REPO, remote_path=REPO_REMOTE, copy=True, ignore=_ignore)
)

hf_secret = modal.Secret.from_name("hf")
checkpoints_volume = modal.Volume.from_name("thunder-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={CHECKPOINT_DIR: checkpoints_volume},
    timeout=3600,
    memory=32768,
)
def infer(config_text: str, ckpt: str, prompt: str, steps: int, target_len: int, profile_len: int | None):
    import sys

    import torch
    import yaml

    sys.path.insert(0, REPO_REMOTE)

    from src.train.diffusion import MaskedDiffusion  # noqa: E402
    from src.train.model import DiffusionLM  # noqa: E402

    cfg = yaml.safe_load(config_text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = f"{CHECKPOINT_DIR}/{ckpt}.pt"

    ckpt_dict = torch.load(ckpt_path, map_location=device)
    base_model = ckpt_dict.get("base_model") or cfg["model"]["base"]
    model = DiffusionLM(base_model).to(device)
    model.load_state_dict(ckpt_dict["model"])
    model.eval()

    dc = cfg["diffusion"]
    diffusion = MaskedDiffusion(
        infer_steps=steps,
        mask_ratio_min=dc.get("mask_ratio_min", 1 / 500),
        mask_ratio_max=dc.get("mask_ratio_max", 1 - 1 / 500),
    )

    print(f"[infer] device={device} | base={base_model} | ckpt={ckpt} "
          f"| steps={steps} | mode=discrete-masked-gen", flush=True)

    def decode(tokens):
        return model.tokenizer.decode(tokens[0], skip_special_tokens=True)

    # ---- behaviour: discrete masked generation (prompt fixed, suffix un-masked progressively) ----
    ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    P = ids.shape[1]
    with torch.no_grad():
        model.generate(ids, diffusion, target_len=target_len)  # warm-up
    with torch.no_grad():
        t0 = time.time()
        suffix = model.generate(ids, diffusion, target_len=target_len)
        dt = time.time() - t0
    generated = decode(suffix)
    n_total = P + suffix.shape[1]
    print(f"[behaviour] prompt ({P} tok): {prompt!r}")
    print(f"[behaviour] generated ({n_total} tok): {prompt + ' ' + generated!r}")
    print(f"[speed] gen_len={n_total} | {n_total / dt:.2f} tok/s | {dt:.2f}s "
          f"for {n_total} tokens @ {steps} steps", flush=True)

    # ---- throughput profile at a different target length (optional) ----
    if profile_len is not None and profile_len != target_len:
        prof_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            model.generate(prof_ids, diffusion, target_len=profile_len)
        with torch.no_grad():
            t0 = time.time()
            suffix = model.generate(prof_ids, diffusion, target_len=profile_len)
            dt = time.time() - t0
        n_total = prof_ids.shape[1] + suffix.shape[1]
        print(f"[profile] target_len={profile_len} | {n_total / dt:.2f} tok/s | {dt:.2f}s "
              f"for {n_total} tokens @ {steps} steps", flush=True)

    return {"device": str(device), "steps": steps, "tokens": n_total}


@app.local_entrypoint()
def run():
    config = os.environ.get("TF_CONFIG", "config/train_config.yaml")
    ckpt = os.environ.get("TF_CKPT", "step_14939")
    prompt = os.environ.get("TF_PROMPT", "Bună ziua, aici este un test de difuzie.")
    steps = int(os.environ.get("TF_STEPS", "24"))
    gpu = os.environ.get("TF_GPU", "").strip() or None
    target_len = int(os.environ.get("TF_TARGET", "256"))
    profile_len = os.environ.get("TF_PROFILE_LEN")
    profile_len = int(profile_len) if profile_len else None

    config_text = (REPO / config).read_text(encoding="utf-8")
    # `.remote()` (not `.spawn()`) blocks and streams the function logs, keeping the
    # app alive until inference returns. `.spawn()` needs `modal run --detach` to
    # survive the local client disconnecting.
    if gpu:
        infer.with_options(gpu=gpu).remote(config_text, ckpt, prompt, steps, target_len, profile_len)
    else:
        infer.remote(config_text, ckpt, prompt, steps, target_len, profile_len)
