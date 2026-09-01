"""One-off diagnostic: why is the training loss ~0?

Loads the real checkpoint from the volume, runs the denoising loss on a fresh batch,
and prints internal magnitudes (x0/pred norms, masked vs unmasked loss split) to tell
whether the model is genuinely denoising, collapsing to a constant, or the masking
branch is not contributing to the loss.

Usage (from repo root), with the HF/wandb secrets and the checkpoint volume:
    TM_DIAG_STEP=3821 modal run infra/diag_loss.py
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = "/vol/checkpoints"
REPO_REMOTE = "/repo"


def _ignore(p: Path) -> bool:
    name = p.name
    if name.startswith("."):
        return True
    if name in {"checkpoints", "__pycache__", "node_modules", ".venv", "venv"}:
        return True
    if name.endswith((".pyc", ".pt", ".log", ".png", ".pdf", ".sqlite", ".sqlite.bak", ".bin")):
        return True
    if p.suffix in {".yaml", ".yml"}:
        return True
    return False


app = modal.App("thunder-fast-diag")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install_from_requirements(str(REPO / "requirements.txt"))
    .apt_install("git")
    .add_local_dir(REPO, remote_path=REPO_REMOTE, copy=True, ignore=_ignore)
)

hf_secret = modal.Secret.from_name("hf")
wandb_secret = modal.Secret.from_name("wandb")
checkpoints_volume = modal.Volume.from_name("thunder-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    secrets=[hf_secret, wandb_secret],
    volumes={CHECKPOINT_DIR: checkpoints_volume},
    timeout=1800,
    memory=32768,
)
def diag(config_text: str, step: int, max_examples: int = 32):
    import sys

    import torch
    import yaml

    sys.path.insert(0, REPO_REMOTE)

    from src.train.data import PackedDataset
    from src.train.diffusion import ContinuousDiffusion
    from src.train.model import DiffusionLM

    cfg = yaml.safe_load(config_text)
    device = torch.device("cuda")
    seq_len = cfg["diffusion"]["seq_len"]

    print(f"[diag] loading Qwen3-0.6B + checkpoint step_{step}.pt ...", flush=True)
    model = DiffusionLM(cfg["model"]["base"]).to(device)
    ckpt = torch.load(f"{CHECKPOINT_DIR}/step_{step}.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    emb = model.word_embeddings.weight.data
    print(f"[diag] hidden={model.hidden_size} | emb std={emb.std().item():.4f} | "
          f"emb row-norm mean={emb.norm(dim=1).mean().item():.4f}", flush=True)

    # A real batch of packed text.
    ds = PackedDataset(
        cfg["data"]["sources"][:1],
        model.tokenizer,
        seq_len=seq_len,
    )
    it = iter(ds)
    batch = torch.stack([next(it) for _ in range(max_examples)]).to(device)
    print(f"[diag] batch {tuple(batch.shape)} | unique tokens: {batch.unique().numel()}", flush=True)

    B, L, D = batch.shape[0], seq_len, model.hidden_size
    x0 = model.embed_tokens(batch)
    t = torch.rand(B, device=device)
    noise = torch.randn_like(x0)
    print(f"[diag] x0 per-token norm mean={x0.norm(dim=-1).mean().item():.4f}", flush=True)

    diff = ContinuousDiffusion(
        hidden_size=model.hidden_size,
        schedule=cfg["diffusion"]["schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        train_steps=cfg["diffusion"]["train_steps"],
        infer_steps=cfg["diffusion"]["infer_steps"],
        prediction=cfg["diffusion"]["prediction"],
        mask_ratio=cfg["diffusion"]["mask_ratio"],
    )

    x_t = diff.q_sample(x0, t, noise)
    use_mask = torch.rand(B, L, device=device) < cfg["diffusion"]["mask_ratio"]
    mask_emb = model.mask_embedding.data.view(1, 1, D).expand(B, L, D).clone()
    x_t = torch.where(use_mask.unsqueeze(-1), mask_emb, x_t)

    pred = model(x_t, t)
    target = diff._prediction_target(x0, noise, t)  # follows cfg["diffusion"]["prediction"]

    mse = (pred - target).pow(2).mean(dim=-1)  # [B, L]
    print(f"[diag] t range=({t.min().item():.3f},{t.max().item():.3f})", flush=True)
    print(f"[diag] ||x0||={x0.norm().item():.4f} ||x_t||={x_t.norm().item():.4f} ||pred||={pred.norm().item():.4f}", flush=True)
    print(f"[diag] ||pred-target||={(pred - target).norm().item():.4f} ||pred-x_t||={(pred - x_t).norm().item():.4f}", flush=True)
    print(f"[diag] frac_masked={use_mask.float().mean().item():.4f}", flush=True)
    print(f"[diag] mse ALL mean={mse.mean().item():.6f}", flush=True)
    if use_mask.any():
        print(f"[diag] mse MASKED  mean={mse[use_mask].mean().item():.6f} (n={int(mse[use_mask].numel())})", flush=True)
    print(f"[diag] mse UNMASKED mean={mse[~use_mask].mean().item():.6f} (n={int(mse[~use_mask].numel())})", flush=True)

    masked_pred = pred[use_mask]
    masked_x0 = x0[use_mask]
    masked_mask_emb = mask_emb[use_mask]
    print(f"[diag] masked pred mean={masked_pred.mean().item():.4f} std={masked_pred.std().item():.4f}", flush=True)
    print(f"[diag] masked x0   mean={masked_x0.mean().item():.4f} std={masked_x0.std().item():.4f}", flush=True)
    print(f"[diag] ||masked pred - mask_emb||={(masked_pred - masked_mask_emb).norm().item():.4f}", flush=True)

    print("[diag] done", flush=True)
    return "diag done"


@app.local_entrypoint()
def run():
    config = os.environ.get("TF_CONFIG", "config/train_config.yaml")
    step = int(os.environ["TF_DIAG_STEP"]) if os.environ.get("TF_DIAG_STEP") else 3821
    max_examples = int(os.environ["TF_DIAG_EXAMPLES"]) if os.environ.get("TF_DIAG_EXAMPLES") else 32
    config_text = (REPO / config).read_text(encoding="utf-8")
    result = diag.remote(config_text, step, max_examples)
    print(f"[launch] diag finished: {result}", flush=True)
