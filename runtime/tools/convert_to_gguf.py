"""Convert a masked-diffusion checkpoint (config.json + diffusion_model.safetensors,
produced by convert/convert_to_diffusion.py) into the runtime's GGUF.

The GGUF carries the hyperparams under `qwen3.*` and the diffusion params under `dlm.*`,
matching runtime/src/engine/model.cpp. Weight keys map HF/DiffusionLM names to ggml names.

Runs in the CI/Modal build environment (requires the `gguf` + `safetensors` packages).
Local dev has no such deps, so this is exercised by the dispatch workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def map_key(k: str) -> str:
    """Map a DiffusionLM state_dict key to a ggml tensor name.

    DiffusionLM wraps the base model: keys are prefixed `base_model.model...`,
    `word_embeddings...`, `lm_head...`. We strip the wrapper to the Qwen3 names.
    """
    k = k.replace("base_model.model.", "blk.", 1)  # partial; layers handled below
    # embeddings / lm_head
    if k.startswith("word_embeddings."):
        return "token_embd.weight"
    if k.startswith("lm_head."):
        return "output.weight"
    if k.startswith("base_model.model.norm."):
        return "output_norm.weight"
    # per-layer
    sep = k.split(".")
    if sep[0].startswith("blk.") and sep[1].isdigit():
        n = int(sep[1])
        kind = ".".join(sep[2:])
        mapping = {
            "self_attn.q_proj.weight": f"blk.{n}.attn_q.weight",
            "self_attn.k_proj.weight": f"blk.{n}.attn_k.weight",
            "self_attn.v_proj.weight": f"blk.{n}.attn_v.weight",
            "self_attn.o_proj.weight": f"blk.{n}.attn_output.weight",
            "mlp.gate_proj.weight":    f"blk.{n}.ffn_gate.weight",
            "mlp.up_proj.weight":      f"blk.{n}.ffn_up.weight",
            "mlp.down_proj.weight":    f"blk.{n}.ffn_down.weight",
            "input_layernorm.weight":  f"blk.{n}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{n}.ffn_norm.weight",
        }
        return mapping.get(kind, k)
    return k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="dir with config.json + diffusion_model.safetensors")
    ap.add_argument("--out", default="runtime.gguf")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text())
    state = load_file(str(ckpt / "diffusion_model.safetensors"))

    import gguf  # requires gguf-py in the build env

    writer = gguf.GGUFWriter(args.out, "qwen3")
    # hyperparams
    writer.add_name(cfg.get("base_model", "thunder-dlm-0.6b"))
    writer.add_uint32("qwen3.context_length", 256)
    writer.add_uint32("qwen3.n_layer", _config_or(cfg, _tensor(state, "base_model.model.layers") , 28))
    # (n_layer/n_embd derived from tensor shapes below)
    writer.add_uint32("qwen3.rms_norm_eps", 1e-6)
    writer.add_uint32("qwen3.rope_freq_base", 10000)
    writer.add_uint32("dlm.mask_token_id", cfg.get("mask_token_id", 151665))
    writer.add_uint32("dlm.infer_steps", cfg.get("infer_steps", 24))

    # weights
    for k, v in state.items():
        gg_name = map_key(k)
        writer.add_tensor(gg_name, v.to(torch.float32).numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {args.out}")


def _tensor(state, prefix: str):
    for k in state:
        if k.startswith(prefix):
            return state[k]
    return None


def _config_or(cfg, tensor, default):
    if tensor is not None:
        return tensor.shape[0] if hasattr(tensor, "shape") else default
    return default


if __name__ == "__main__":
    main()
