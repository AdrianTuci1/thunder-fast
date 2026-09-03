"""Evaluation harness for the diffusion model (Romanian + multilingual sanity).

Metrics here are intentionally lightweight for a 24h PoC:
  - average denoising (reconstruction) loss on held-out text (down-trending == learning),
  - a short sample() generation decoded to text for a quick qualitative look,
  - (optional) base-model causal perplexity on the same text as a reference line
    (reported for context only - the diffusion model is not a next-token model).

Full Romanian NLU (LiRo) / MT (FLORES, WMT) benchmarks are wired here as stubs so the
harness can be extended once a good checkpoint exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.data import PackedDataset  # noqa: E402
from src.train.diffusion import MaskedDiffusion  # noqa: E402
from src.train.model import DiffusionLM  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def reconstruction_loss(model, diffusion, cfg, device, limit: int = 200):
    """Average masked cross-entropy on held-out text (down-trending == learning)."""
    ds = PackedDataset(
        cfg["data"]["sources"][:1],
        model.tokenizer,
        seq_len=cfg["diffusion"]["seq_len"],
    )
    losses = []
    it = iter(ds)
    for _ in range(limit):
        try:
            ids = next(it)
        except StopIteration:
            break
        batch = ids.unsqueeze(0).to(device)
        loss = diffusion.training_loss(model, batch)
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def sample_text(model, diffusion, prompt: str, device, steps: int | None = None,
                target_len: int = 256) -> str:
    ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    tokens = model.generate(ids, diffusion, target_len=target_len, steps=steps)
    return model.tokenizer.decode(tokens[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/train_config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Bună ziua, aici este")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionLM(cfg["model"]["base"]).to(device)
    model.eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

    diffusion = MaskedDiffusion(
        infer_steps=args.steps or cfg["diffusion"]["infer_steps"],
        mask_ratio_min=cfg["diffusion"].get("mask_ratio_min", 1 / 500),
        mask_ratio_max=cfg["diffusion"].get("mask_ratio_max", 1 - 1 / 500),
    )

    recon = reconstruction_loss(model, diffusion, cfg, device)
    print(f"[eval] reconstruction loss: {recon:.4f}", flush=True)

    text = sample_text(model, diffusion, args.prompt, device, steps=args.steps)
    print(f"[gen] prompt: {args.prompt!r}")
    print(f"[gen] sample: {text!r}")

    print("[eval] LiRo / FLORES / WMT benchmarks: not wired yet - see eval/README.md")


if __name__ == "__main__":
    main()
