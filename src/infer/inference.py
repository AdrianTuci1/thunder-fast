"""PyTorch inference for the diffusion model (the "reference" path).

Loads a trained diffusion checkpoint, runs the reverse-diffusion loop (`sample`) for the
fixed output length (seq_len = 256 tokens), decodes to text, and reports throughput
(tokens/sec). This is the in-repo reference implementation used to sanity-check quality and
speed BEFORE the custom ggml CPU runtime (ADR 0003) is built.

Usage:
    python src/infer/inference.py --ckpt <path.pt> --prompt "Bună ziua, eu sunt"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train.diffusion import ContinuousDiffusion  # noqa: E402
from src.train.model import DiffusionLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-model", default=None, help="defaults to the ckpt's base_model")
    ap.add_argument("--prompt", default="Bună ziua, aici este un test de difuzie.")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--schedule", default="linear")
    ap.add_argument("--prediction", default="x0")
    ap.add_argument("--mask-ratio", type=float, default=0.25)
    ap.add_argument("--beta-start", type=float, default=0.0001)
    ap.add_argument("--beta-end", type=float, default=0.02)
    ap.add_argument("--gpu", action="store_true", help="use CUDA (default CPU)")
    args = ap.parse_args()

    device = torch.device("cuda" if args.gpu else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    base_model = args.base_model or ckpt.get("base_model", "Qwen/Qwen3-0.6B")

    model = DiffusionLM(base_model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = ContinuousDiffusion(
        hidden_size=model.hidden_size,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        train_steps=args.steps,
        infer_steps=args.steps,
        prediction=args.prediction,
        mask_ratio=args.mask_ratio,
    )

    ids = model.tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"prompt ({ids.shape[1]} tokens): {args.prompt!r}", flush=True)

    # Warm-up + timing.
    with torch.no_grad():
        embeds = model.sample(ids, diffusion)
    t0 = time.time()
    with torch.no_grad():
        embeds = model.sample(ids, diffusion)
    dt = time.time() - t0

    tokens = model.decode_embeddings_to_tokens(embeds)
    text = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
    # Output length is fixed to seq_len (256) by the denoising loop.
    out_tokens = tokens.shape[1]

    print(f"tokens/sec: {out_tokens / dt:.2f}  ({dt:.2f}s for {out_tokens} tokens, {args.steps} steps)")
    print(f"output ({out_tokens} tokens): {text!r}")


if __name__ == "__main__":
    main()
