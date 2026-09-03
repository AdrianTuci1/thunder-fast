"""Turn an autoregressive (causal) Hugging Face LLM into a discrete masked-diffusion denoiser.

A masked-diffusion adaptation requires three structural changes to an existing AR
checkpoint:
  1. causal attention -> bidirectional (full) attention,
  2. a discrete `[MASK]` token added to the vocabulary (the "noise" token),
  3. re-targeting the output head at masked-token reconstruction (cross-entropy over the
     vocabulary) rather than next-token generation.

We keep the original weights and tokenizer and wrap the base transformer. There is no
continuous noise schedule and no learned mask embedding / time conditioning: the model
simply predicts logits at every position, and only the masked positions contribute loss.
"""

from __future__ import annotations

import os

# HF downloads from Modal containers sometimes time out on the HEAD (etag) request at the
# default 10s. Be more patient so model loading is robust against transient network slowness.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def make_bidirectional(model) -> None:
    """Put the transformer on the SDPA path and request non-causal attention.

    CAUTION: setting `module.is_causal = False` alone does NOT make the decoder bidirectional.
    HF's `_update_causal_mask` builds a *causal* 4-D mask whenever `attention_mask` is None (or
    2-D), and that mask overrides the module's `is_causal`. The actual bidirectional behaviour
    must be enforced at forward time by passing an explicit 4-D all-zeros mask (see
    `DiffusionLM.forward`). We keep the fast SDPA (FlashAttention) kernel path; "eager" would run
    attention as slow FP32 matmuls and kill GPU throughput. NOTE: must be validated on the exact
    base architecture (see AGENTS.md) - HF versions vary.
    """
    for module in model.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
        if hasattr(module, "config"):
            module.config._attn_implementation = "sdpa"
    return model


def _sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    alg: str = "origin",
):
    """Sampling + a confidence metric over logits [..., V].

    Returns (confidence, token_ids) both shaped like logits without the vocab dim.
    With temperature=0 this is argmax (deterministic); otherwise it multinomial-samples.
    `alg` selects the confidence metric used for progressive unmasking (entropy is the
    default; `topk_margin` is the prob gap between top-1 and top-2).
    """
    if temperature > 0:
        logits = logits / temperature
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        idx = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(idx, torch.finfo(logits.dtype).min)
    if top_p is not None and top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

    probs = torch.softmax(logits.float(), dim=-1)
    if temperature > 0:
        # torch.multinomial only accepts 1-D/2-D; flatten leading dims then restore.
        flat = probs.reshape(-1, probs.size(-1))
        x0 = torch.multinomial(flat, 1).squeeze(-1).reshape(probs.shape[:-1])
    else:
        x0 = probs.argmax(dim=-1)
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if alg == "topk_margin":
        top2 = probs.topk(min(2, probs.size(-1)), dim=-1).values
        confidence = top2[..., 0] - top2[..., 1]
    elif alg == "entropy":
        log_p = torch.log(probs.clamp(min=1e-10))
        confidence = (probs * log_p).sum(dim=-1)

    return confidence, x0


def _discrete_generate_window(
    forward,
    mask_token_id: int,
    prompt_ids: torch.Tensor,
    target_len: int,
    steps: int,
    temperature: float,
    top_p: float | None,
    top_k: int,
    alg: str,
    eps: float,
    device: torch.device,
    alg_temp: float | None = None,
) -> torch.Tensor:
    """Progressive discrete un-masking over a single fixed window.

    `forward(input_ids) -> logits [B, L, V]` is any bidirectional masked-diffusion model.
    Prompt tokens are held fixed; the remaining positions start from `mask_token_id` and are
    revealed most-confident-first. Returns the generated suffix token ids [B, G].
    """
    B = prompt_ids.shape[0]
    G = max(1, target_len - prompt_ids.shape[1])
    x = torch.cat(
        [prompt_ids, torch.full((B, G), mask_token_id, dtype=torch.long, device=device)], dim=1
    )
    fix_mask = x != mask_token_id  # prompt / already-generated positions are never re-masked
    timesteps = torch.linspace(1.0, eps, steps + 1, device=device)

    for i in range(steps):
        mask_index = x == mask_token_id
        if not mask_index.any():
            break
        logits = forward(x)
        # Next-token alignment (matches training): position i predicts token i+1.
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        t, s = timesteps[i], timesteps[i + 1]

        if alg == "origin":
            p_transfer = (1.0 - s / t) if i < steps - 1 else 1.0
            transfer = mask_index & (torch.rand_like(x.float()) < p_transfer)
            _, cand_tok = _sample_tokens(logits[transfer], temperature, top_p, top_k, alg)
            x[transfer] = cand_tok
        elif alg == "p2":
            conf, x0_full = _sample_tokens(logits, temperature, top_p, top_k, "entropy")
            full_conf = conf.clone()
            full_conf[fix_mask] = float("inf")  # never re-mask fixed positions
            num_pos = (~fix_mask).sum(dim=1)
            num_to_mask = (num_pos.float() * (1.0 - (i + 1) / steps)).floor().long()
            num_to_mask = num_to_mask.clamp_min(0).clamp_max(num_pos)
            max_k = int(num_to_mask.max().item())
            to_mask = torch.zeros_like(x, dtype=torch.bool)
            if max_k > 0:
                sorted_idx = torch.argsort(full_conf, dim=1, descending=False)
                topk_idx = sorted_idx[:, :max_k]
                row_ok = torch.arange(max_k, device=device).unsqueeze(0) < num_to_mask.unsqueeze(1)
                bi = torch.arange(B, device=device).unsqueeze(1).expand_as(topk_idx)[row_ok]
                ci = topk_idx[row_ok]
                to_mask[bi, ci] = True
            x[to_mask] = mask_token_id
            keep = mask_index & (~to_mask)
            x[keep] = x0_full[keep]
        else:  # confidence-based: entropy / maskgit_plus / topk_margin
            conf, x0_full = _sample_tokens(logits, temperature, top_p, top_k, alg)
            num_masked = mask_index.sum(dim=1)
            n_transfer = (
                (num_masked.float() * (1.0 - s / t)).long()
                if i < steps - 1
                else num_masked
            )
            max_tr = int(n_transfer.max().item())
            if max_tr > 0:
                full_conf = torch.full_like(x, -torch.inf, dtype=conf.dtype, device=device)
                full_conf[mask_index] = conf[mask_index]
                if alg_temp is not None and alg_temp > 0:
                    # Soften the confidence into an unmask-position distribution (`alg_temp`,
                    # default 0.6) and sample which positions to transfer.
                    unmask_probs = F.softmax(full_conf / alg_temp, dim=-1)
                    top_idx = torch.multinomial(unmask_probs, num_samples=max_tr, replacement=False)
                else:
                    _, top_idx = full_conf.topk(max_tr, dim=1)
                row_ok = torch.arange(max_tr, device=device).unsqueeze(0) < n_transfer.unsqueeze(1)
                bi = torch.arange(B, device=device).unsqueeze(1).expand_as(top_idx)[row_ok]
                ci = top_idx[row_ok]
                src = torch.full_like(x, mask_token_id)
                src[mask_index] = x0_full[mask_index]
                x[bi, ci] = src[bi, ci]

    return x[:, prompt_ids.shape[1]:]


def _discrete_generate_long(
    forward,
    mask_token_id: int,
    eos_token_id: int | None,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    block_len: int,
    steps: int,
    stop_at_eos: bool,
    device: torch.device,
    **gen_kwargs,
) -> torch.Tensor:
    """Chained block generation (our long-output engine): one window per pass, growing context.

    Generates `block_len` tokens per diffusion pass, appends them to the context (prompt +
    everything generated so far), then generates the next block. Returns the full sequence
    [B, prompt_len + generated]. Stops early if `stop_at_eos` and the last block contains EOS.
    """
    gen: list[torch.Tensor] = []
    produced = 0
    while produced < max_new_tokens:
        n = min(block_len, max_new_tokens - produced)
        ctx = torch.cat([prompt_ids] + gen, dim=1)  # [B, P + produced]
        block = _discrete_generate_window(
            forward, mask_token_id, ctx, ctx.shape[1] + n, steps,
            temperature=gen_kwargs.pop("temperature", 0.0),
            top_p=gen_kwargs.pop("top_p", None),
            top_k=gen_kwargs.pop("top_k", 200),
            alg=gen_kwargs.pop("alg", "entropy"),
            eps=gen_kwargs.pop("eps", 1e-3),
            device=device,
            alg_temp=gen_kwargs.pop("alg_temp", None),
        )
        gen.append(block)
        produced += n
        if stop_at_eos and eos_token_id is not None and (block == eos_token_id).any():
            break
    return torch.cat([prompt_ids] + gen, dim=1)


class DiffusionLM(nn.Module):
    def __init__(
        self,
        base_model_id: str,
        use_cache: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        # A discrete mask token is the crux of MDM. Add it if the base tokenizer lacks one.
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})

        config = AutoConfig.from_pretrained(base_model_id)
        # SDPA (FlashAttention) for GPU throughput; causal masking is disabled by is_causal.
        config._attn_implementation = "sdpa"
        config.use_cache = False
        # Set `use_cache`/attn on the *config*, not as `from_pretrained` kwargs: recent
        # transformers (4.51+) forwards unknown kwargs to the model __init__, and Qwen2's
        # constructor (Qwen2ForCausalLM) does not accept `use_cache`.
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, config=config
        )
        make_bidirectional(self.base_model)
        # Resize to include the new [MASK] row (grows both embed_tokens and lm_head).
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.backbone = self.base_model.model  # transformer stack (without lm_head/embedding)
        self.lm_head = self.base_model.lm_head
        self.word_embeddings = self.base_model.get_input_embeddings()
        self.mask_token_id = self.tokenizer.mask_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Denoise. `input_ids` = [B, L] token ids (masked positions carry `mask_token_id`).

        Returns logits [B, L, V] under full BIDIRECTIONAL attention.

        We must pass an explicit 4-D mask, not `attention_mask=None`. HF's `_update_causal_mask`
        builds a *causal* 4-D mask whenever `attention_mask` is None (or 2-D), and that mask
        overrides the `is_causal=False` set by `make_bidirectional`, silently leaving the model
        causal. A 4-D all-zeros (bool, all-unmasked) mask is used directly by the SDPA path and
        makes every position attend to every position — the MDM denoising mode the model must run.
        """
        backbone = self.backbone
        L = input_ids.shape[1]
        attn_mask = torch.zeros((1, 1, L, L), dtype=torch.bool, device=input_ids.device)
        if input_ids.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = backbone(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
        else:
            out = backbone(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
        return self.lm_head(out.hidden_states[-1])

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    # ---- inference: discrete progressive unmasking over one window, or chained blocks ----
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        diffusion,
        target_len: int = 256,
        steps: int | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        top_k: int = 200,
        alg: str = "entropy",
        eps: float = 1e-3,
        alg_temp: float | None = None,
    ) -> torch.Tensor:
        """Generate `target_len - prompt_len` tokens in ONE fixed window (LLaDA-style).

        The prompt tokens are held fixed (clean, fully observed); the remaining positions
        start from `mask_token_id` and are progressively un-masked, most-confident first.
        Returns the generated token ids, shape [B, G] (G = target_len - prompt_len).
        """
        steps = steps or diffusion.infer_steps
        device = next(self.parameters()).device
        return _discrete_generate_window(
            self.forward, self.mask_token_id, prompt_ids, target_len, steps,
            temperature, top_p, top_k, alg, eps, device, alg_temp=alg_temp,
        )

    @torch.no_grad()
    def generate_long(
        self,
        prompt_ids: torch.Tensor,
        diffusion,
        max_new_tokens: int = 2048,
        block_len: int = 256,
        steps: int | None = None,
        stop_at_eos: bool = True,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Chained block generation for outputs longer than one window (up to +2048).

        Generates `block_len` tokens per diffusion pass, appends them to the growing
        context (prompt + all previously generated), then generates the next block. Returns
        the full sequence [B, prompt_len + generated]. Stops early if `stop_at_eos` and the
        last block contains the EOS token.
        """
        steps = steps or diffusion.infer_steps
        device = next(self.parameters()).device
        eos = self.tokenizer.eos_token_id
        return _discrete_generate_long(
            self.forward, self.mask_token_id, eos, prompt_ids, max_new_tokens,
            block_len, steps, stop_at_eos, device, **gen_kwargs,
        )

    def set_train_ctx(self):
        self.base_model.train()
