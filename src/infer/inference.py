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

from src.train.diffusion import MaskedDiffusion  # noqa: E402
from src.train.model import DiffusionLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-model", default=None, help="defaults to the ckpt's base_model")
    ap.add_argument("--prompt", default="Bună ziua, aici este un test de difuzie.")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--target", type=int, default=256, help="generated tokens (one window)")
    ap.add_argument("--gpu", action="store_true", help="use CUDA (default CPU)")
    args = ap.parse_args()

    device = torch.device("cuda" if args.gpu else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    base_model = args.base_model or ckpt.get("base_model", "Qwen/Qwen2-0.5B")

    model = DiffusionLM(base_model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = MaskedDiffusion(infer_steps=args.steps)

    ids = model.tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"prompt ({ids.shape[1]} tokens): {args.prompt!r}", flush=True)

    # Warm-up + timing.
    with torch.no_grad():
        model.generate(ids, diffusion, target_len=args.target, steps=args.steps)
    t0 = time.time()
    with torch.no_grad():
        tokens = model.generate(ids, diffusion, target_len=args.target, steps=args.steps)
    dt = time.time() - t0

    text = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
    # Output length is fixed to the generation window (seq_len = 256 by default).
    out_tokens = tokens.shape[1]

    print(f"tokens/sec: {out_tokens / dt:.2f}  ({dt:.2f}s for {out_tokens} tokens, {args.steps} steps)")
    print(f"output ({out_tokens} tokens): {text!r}")


if __name__ == "__main__":
    main()
