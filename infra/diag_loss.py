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
    from src.train.diffusion import MaskedDiffusion
    from src.train.model import DiffusionLM

    cfg = yaml.safe_load(config_text)
    device = torch.device("cuda")
    seq_len = cfg["diffusion"]["seq_len"]

    print(f"[diag] loading Qwen3-0.6B + checkpoint step_{step}.pt ...", flush=True)
    model = DiffusionLM(cfg["model"]["base"]).to(device)
    model.eval()
    try:
        ckpt = torch.load(f"{CHECKPOINT_DIR}/step_{step}.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"[diag] WARNING: checkpoint incompatible (objective changed), using base model: {e}", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)

    emb = model.word_embeddings.weight.data
    print(f"[diag] hidden={model.hidden_size} | vocab={model.vocab_size} | "
          f"mask_token_id={model.mask_token_id} | emb std={emb.std().item():.4f}", flush=True)

    # A real batch of packed text.
    ds = PackedDataset(
        cfg["data"]["sources"][:1],
        model.tokenizer,
        seq_len=seq_len,
    )
    it = iter(ds)
    batch = torch.stack([next(it) for _ in range(max_examples)]).to(device)
    print(f"[diag] batch {tuple(batch.shape)} | unique tokens: {batch.unique().numel()}", flush=True)

    diff = MaskedDiffusion(
        infer_steps=cfg["diffusion"].get("infer_steps", 24),
        mask_ratio_min=cfg["diffusion"].get("mask_ratio_min", 1 / 500),
        mask_ratio_max=cfg["diffusion"].get("mask_ratio_max", 1 - 1 / 500),
    )

    # Discrete MDM: mask a random per-sample fraction, run the model, measure CE + accuracy
    # at masked positions. A healthy adapted model has masked-token accuracy well above the
    # vocab-size baseline and a CE that decreases. (Collapse would show near-random accuracy.)
    r = diff._sample_mask_ratio(batch.shape[0], device)
    use_mask = torch.rand_like(batch.float()) < r.unsqueeze(-1)
    x_m = torch.where(use_mask, torch.full_like(batch, model.mask_token_id), batch)
    logits = model(x_m)
    pred_logits = logits[..., :-1, :].contiguous()
    labels = batch[..., 1:].contiguous()
    valid = use_mask[..., 1:]

    token_loss = torch.nn.functional.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(valid)
    valid_any = valid.any()
    mdm_loss = (token_loss * valid).sum() / (valid.sum() + 1e-8)

    pred_tok = pred_logits.argmax(dim=-1)
    acc_masked = (pred_tok[valid] == labels[valid]).float().mean().item() if valid_any else float("nan")
    acc_all = (pred_tok == labels).float().mean().item()

    # Random baseline for a vocab of this size.
    base_acc = 1.0 / model.vocab_size

    print(f"[diag] mask_ratio range=({r.min().item():.3f},{r.max().item():.3f}) frac_masked={use_mask.float().mean().item():.4f}", flush=True)
    print(f"[diag] mdm_loss(loss*full)={mdm_loss.item():.4f}", flush=True)
    print(f"[diag] masked_token_acc={acc_masked:.4f} (random={base_acc:.6f}) | all_acc={acc_all:.4f}", flush=True)

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
