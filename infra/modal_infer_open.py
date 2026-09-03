"""Run diffusion inference on a masked-diffusion checkpoint through OUR engine.

Loads a masked-diffusion `Qwen2ForCausalLM` (the modeling lives in-repo at `infra/qwen_ref`, with
no-op stubs for its distributed/data imports), then generates text with OUR block-wise
discrete-diffusion engine (`_discrete_generate_long`): 256-token windows, 24 unmask steps, up to
+2048 tokens. Also supports a single large window (`single`).

Why the bundled modeling instead of a checkpoint's own module: the shipped `modeling_qwen2.py` in
some checkpoints has `_sample`/`_prepare_generation_config` that collide with transformers'
`GenerationMixin` in the MRO, and its `_sample` calls `multinomial` with `n_sample <= 0` at late
steps. We drive the bundled, clean forward instead.

Crucial correctness point: a MASKED-DIFFUSION model MUST run with full BIDIRECTIONAL attention.
`_update_causal_mask` builds a *causal* 4-D mask whenever `attention_mask` is None (or 2-D), so
passing `is_causal=False` alone is a no-op. We therefore pass an explicit 4-D all-zeros mask, which
makes the attention kernel attend to every position. Running causal produces degenerate, repeated
garbage; bidirectional produces coherent code.

Usage (from the repo root):
    TF_PROMPT="Write a quick sort algorithm in python." TF_MAX_NEW_TOKENS=2048 modal run infra/modal_infer_open.py
    TF_MODE=single  # single 2048-token window instead of block-wise
    TF_GPU=A10G
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = "/vol/checkpoints"
REPO_REMOTE = "/repo"
MODEL_DIR = f"{CHECKPOINT_DIR}/ref-mdm-0.5B"


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


app = modal.App("thunder-fast-infer-open")

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
def infer(model_dir: str, prompt: str, max_new_tokens: int, block_len: int, steps: int,
          mode: str, temperature: float, top_k: int, alg: str, alg_temp: float,
          stop_at_eos: bool):
    import sys

    import torch

    sys.path.insert(0, REPO_REMOTE)
    sys.path.insert(0, f"{REPO_REMOTE}/infra")

    # OUR engine: the block-wise discrete-diffusion generator.
    from src.train.model import _discrete_generate_long, _discrete_generate_window

    # Neutralize `replace_return_docstrings` (version-agnostic; we only call `forward`).
    import transformers.utils as _tu
    from transformers.utils import doc as _doc

    def _identity_docstrings(output_type=None, config_class=None):
        return lambda fn: fn

    _tu.replace_return_docstrings = _identity_docstrings
    _doc.replace_return_docstrings = _identity_docstrings

    # Load the bundled Qwen2 MDM modeling (the clean, self-contained one).
    from qwen_ref.modeling_qwen2 import Qwen2ForCausalLM
    from qwen_ref.generation_utils import MDMGenerationConfig

    from transformers import AutoConfig, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    # The forward uses `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]` for non-eager;
    # most checkpoints ship `eager`, which is the bundled clean eager_attention_forward.
    if config._attn_implementation is None:
        config._attn_implementation = "eager"
    model = Qwen2ForCausalLM.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    tied = torch.equal(model.model.embed_tokens.weight, model.lm_head.weight)
    print(f"[model] params={n_params} | tie_embed_lmhead={tied} | "
          f"pad_token_id={model.config.pad_token_id} | attn={config._attn_implementation}",
          flush=True)

    mask_token_id = tokenizer.mask_token_id
    eos_token_id = model.config.eos_token_id
    if mask_token_id is None:
        raise ValueError("tokenizer has no mask_token_id; downloaded repo is not a valid MDM tokenizer")

    # THE FIX: the model must run with full BIDIRECTIONAL attention. `_update_causal_mask`
    # emits a *causal* 4-D mask whenever `attention_mask` is None (or 2-D), so `is_causal=False`
    # alone is a no-op. Passing an explicit 4-D all-zeros mask makes it use that directly, so every
    # position attends to every position (the MDM denoising mode this checkpoint was trained in).
    def forward_bidir(x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        m = torch.zeros((1, 1, L, L), device=device)
        return model(input_ids=x, attention_mask=m, is_causal=False).logits

    # Causal variant (diagnostic only): additive upper-triangular -inf mask.
    def forward_causal(x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        min_dtype = torch.finfo(torch.float32).min
        m = torch.triu(torch.full((L, L), min_dtype, device=device), diagonal=1)[None, None]
        return model(input_ids=x, attention_mask=m, is_causal=False).logits

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]
    print(f"[infer] device={device} | model_dir={model_dir} | mode={mode} | steps={steps} "
          f"| max_new={max_new_tokens} | block_len={block_len} | alg={alg} "
          f"| temp={temperature} | top_k={top_k} | alg_temp={alg_temp} | stop_at_eos={stop_at_eos} "
          f"| mask_token_id={mask_token_id}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        if mode == "single":
            suffix = _discrete_generate_window(
                forward_bidir, mask_token_id, ids, ids.shape[1] + max_new_tokens, steps,
                temperature, None, top_k, alg, 1e-3, device, alg_temp=alg_temp,
            )
            out = torch.cat([ids, suffix], dim=1)
        elif mode == "ar":
            # Greedy autoregressive decode under CAUSAL attention — a sanity check on the weights.
            # Runs the full growing sequence each step (no KV-cache), so it is a lower bound on AR speed.
            cur = ids
            for _ in range(max_new_tokens):
                tok = forward_causal(cur)[0, -1].argmax(-1).view(1, 1)
                cur = torch.cat([cur, tok], dim=1)
            out = cur
        elif mode == "native":
            # The bundled `diffusion_generate` sampling loop. It passes attention_mask=None, so it
            # runs CAUSAL and is expected to be garbage — kept only to contrast with our
            # bidirectional engine.
            cfg = MDMGenerationConfig(
                mask_token_id=mask_token_id, pad_token_id=model.config.pad_token_id,
                eos_token_id=eos_token_id, max_new_tokens=max_new_tokens, steps=steps,
                temperature=temperature, top_k=top_k, alg=alg, alg_temp=alg_temp,
                num_return_sequences=1, return_dict_in_generate=True,
            )
            out = model.diffusion_generate(inputs=ids, generation_config=cfg).sequences
        else:  # block-wise (our long-output engine)
            out = _discrete_generate_long(
                forward_bidir, mask_token_id, eos_token_id, ids, max_new_tokens, block_len,
                steps, stop_at_eos, device, temperature=temperature, top_p=None, top_k=top_k,
                alg=alg, eps=1e-3, alg_temp=alg_temp,
            )
    dt = time.time() - t0

    gen = out[0][prompt_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    n_out = out.shape[1]
    print(f"[speed] {n_out / dt:.2f} tok/s | {dt:.2f}s for {n_out} tokens @ {steps} steps", flush=True)
    print(f"[prompt] ({prompt_len} tok): {prompt!r}", flush=True)
    print(f"[output] ({n_out - prompt_len} tok): {text!r}", flush=True)

    return {"device": str(device), "mode": mode, "tokens": n_out}


@app.local_entrypoint()
def run():
    prompt = os.environ.get("TF_PROMPT", "Write a quick sort algorithm in python.")
    max_new_tokens = int(os.environ.get("TF_MAX_NEW_TOKENS", "2048"))
    block_len = int(os.environ.get("TF_BLOCK_LEN", "256"))
    steps = int(os.environ.get("TF_STEPS", "24"))
    mode = os.environ.get("TF_MODE", "block")
    temperature = float(os.environ.get("TF_TEMPERATURE", "0.7"))
    top_k = int(os.environ.get("TF_TOP_K", "500"))
    alg = os.environ.get("TF_ALG", "entropy")
    alg_temp = float(os.environ.get("TF_ALG_TEMP", "0.6"))  # unmask-position confidence softening
    stop_at_eos = os.environ.get("TF_STOP_AT_EOS", "false").lower() in {"1", "true", "yes"}
    gpu = os.environ.get("TF_GPU", "").strip() or None
    model_dir = os.environ.get("TF_MODEL_DIR", MODEL_DIR)

    args = (model_dir, prompt, max_new_tokens, block_len, steps, mode, temperature, top_k,
            alg, alg_temp, stop_at_eos)
    if gpu:
        infer.with_options(gpu=gpu).remote(*args)
    else:
        infer.remote(*args)
