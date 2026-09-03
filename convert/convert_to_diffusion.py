"""Build a masked-diffusion-adapted checkpoint from a pretrained autoregressive LLM.

This produces the *initial* discrete MDM weights: the base AR weights plus a newly added
`[MASK]` token row in the embedding matrix and LM head, so training has a clean start.
It writes:

    model/config.json                       diffusion config (bidirectional flag, mask_token_id, ...)
    model/diffusion_model.safetensors       the full adapted weights

NOTE: exporting to GGUF / the custom ggml runtime (the real "binary" for CPU inference)
is a separate downstream step that we build later (ADR 0003). This script is the
training-side conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="model")
    ap.add_argument("--infer-steps", type=int, default=24)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.train.model import DiffusionLM

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionLM(args.base_model).to(dev)
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    state = {k: v.to("cpu") for k, v in model.state_dict().items()}
    save_file(state, out / "diffusion_model.safetensors")

    cfg = {
        "base_model": args.base_model,
        "arch": "discrete_diffusion_llm",
        "bidirectional_attention": True,
        "mask_token_id": model.mask_token_id,
        "infer_steps": args.infer_steps,
        "hidden_size": model.hidden_size,
        "vocab_size": model.vocab_size,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Wrote diffusion checkpoint to {out}/ (config.json + diffusion_model.safetensors)")


if __name__ == "__main__":
    main()
