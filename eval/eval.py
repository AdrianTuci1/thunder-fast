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
from src.train.diffusion import ContinuousDiffusion  # noqa: E402
from src.train.model import DiffusionLM  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def reconstruction_loss(model, diffusion, cfg, device, limit: int = 200):
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
        x0 = model.embed_tokens(batch)
        t = torch.rand(batch.shape[0], device=device)
        loss = diffusion.training_loss(model, x0, model.mask_embedding.data, t)
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def sample_text(model, diffusion, prompt: str, device, steps: int | None = None) -> str:
    ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    embeds = model.sample(ids, diffusion, steps=steps)
    tokens = model.decode_embeddings_to_tokens(embeds)
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

    diffusion = ContinuousDiffusion(
        hidden_size=model.hidden_size,
        schedule=cfg["diffusion"]["schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        train_steps=cfg["diffusion"]["train_steps"],
        infer_steps=args.steps or cfg["diffusion"]["infer_steps"],
        prediction=cfg["diffusion"]["prediction"],
        mask_ratio=cfg["diffusion"]["mask_ratio"],
    )

    recon = reconstruction_loss(model, diffusion, cfg, device)
    print(f"[eval] reconstruction loss: {recon:.4f}", flush=True)

    text = sample_text(model, diffusion, args.prompt, device, steps=args.steps)
    print(f"[gen] prompt: {args.prompt!r}")
    print(f"[gen] sample: {text!r}")

    print("[eval] LiRo / FLORES / WMT benchmarks: not wired yet - see eval/README.md")


if __name__ == "__main__":
    main()
