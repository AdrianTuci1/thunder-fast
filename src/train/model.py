"""Turn an autoregressive (causal) Hugging Face LLM into a continuous-diffusion denoiser.

DiffuLLaMA-style adaptation requires three structural changes to an existing AR checkpoint:
  1. causal attention -> bidirectional (full) attention,
  2. injected diffusion timestep conditioning,
  3. an output head that predicts the clean embedding (x0 / noise / v) rather than next tokens.

We keep the original weights and tokenizer, wrap the base transformer, and add a small
time-conditioning MLP plus a learned mask embedding. The model produces hidden-space
embeddings (pre-LM-head); token recovery at inference is done by nearest-neighbour lookup
or by passing the denoised embedding through the original lm_head.
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
    """Disable causal masking on every attention module in the model.

    Each attention head exposes an `is_causal` attribute; forcing it to False makes the
    decoder fully bidirectional once no causal mask is passed at forward time. We keep the
    model on the fast SDPA kernel path (FlashAttention) - switching to "eager" would run the
    attention as slow FP32 matmuls and kill throughput on GPU. NOTE: must be validated on
    the exact base architecture (see AGENTS.md) - HF versions vary.
    """
    for module in model.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
        if hasattr(module, "config"):
            module.config._attn_implementation = "sdpa"
    return model


class DiffusionLM(nn.Module):
    def __init__(
        self,
        base_model_id: str,
        time_emb_dim: int | None = None,
        use_cache: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        config = AutoConfig.from_pretrained(base_model_id)
        # SDPA (FlashAttention) for GPU throughput; causal masking is disabled by is_causal.
        config._attn_implementation = "sdpa"
        config.use_cache = False
        # Set `use_cache`/attn on the *config*, not as `from_pretrained` kwargs: recent
        # transformers (4.51+) forwards unknown kwargs to the model __init__, and Qwen3's
        # constructor (Qwen3ForCausalLM) does not accept `use_cache`.
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, config=config
        )
        make_bidirectional(self.base_model)

        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.backbone = self.base_model.model  # transformer stack (without lm_head/embedding)
        self.lm_head = self.base_model.lm_head
        self.word_embeddings = self.base_model.get_input_embeddings()

        time_emb_dim = time_emb_dim or self.hidden_size
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, self.hidden_size),
        )
        # Learned mask embedding (absorbing state for the masking branch).
        self.mask_embedding = nn.Parameter(
            torch.randn(self.hidden_size) * 0.02
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        """Denoise. `x_t` = [B, L, hidden] noisy/masked embeddings, `t` = [B] in [0,1].

        Returns the predicted clean embedding (parameterized as x0/epsilon/v by the
        diffusion object) in hidden space, shape [B, L, hidden].
        """
        B, L, D = x_t.shape
        t = t.float().view(B, 1)
        time_token = self.time_mlp(t).unsqueeze(1)  # [B, 1, D]
        h = x_t + time_token
        # Fully bidirectional: the causal mask is disabled (is_causal=False, see
        # make_bidirectional) and there is no padding (all 256 tokens attend), so no
        # attention_mask is passed. bf16 autocast uses the H100 tensor cores.
        if x_t.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.backbone(inputs_embeds=h, attention_mask=None, output_hidden_states=True)
        else:
            out = self.backbone(inputs_embeds=h, attention_mask=None, output_hidden_states=True)
        # Return in fp32 so the diffusion loss is not computed in a ragged dtype.
        return out.hidden_states[-1].float()  # [B, L, D]

    # ---- inference: reverse diffusion over `steps`, given a Diffusion object ----
    @torch.no_grad()
    def sample(
        self,
        prompt_ids: torch.Tensor,
        diffusion,
        steps: int | None = None,
        temperature: float = 1.0,
    ):
        """Reverse-diffusion loop (DDPM ancestral) over the prompt embeddings.

        Starts from the noised version of the prompt at t=1 and denoises to t=0.
        Returns denoised embeddings; call `decode_embeddings_to_tokens` to get token ids.
        """
        steps = steps or diffusion.infer_steps
        device = next(self.parameters()).device
        B, L = prompt_ids.shape
        seq = self.word_embeddings(prompt_ids)  # [B, L, D] clean (content to condition on)
        x = diffusion.q_sample(seq, torch.full((B,), 1.0, device=device), torch.randn_like(seq))

        t_levels = diffusion.inference_schedule(device)  # descending, length == steps
        for i, t_cur in enumerate(t_levels):
            t = t_cur.expand(B)
            pred_x0 = self.forward(x, t)
            if diffusion.prediction == "epsilon":
                abar = diffusion._alpha_bar(t).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
                pred_x0 = (x - (1.0 - abar).clamp(min=0).sqrt() * pred_x0) / abar.sqrt()
            elif diffusion.prediction == "v":
                abar = diffusion._alpha_bar(t).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
                sa, sm = abar.sqrt(), (1.0 - abar).clamp(min=0).sqrt()
                pred_x0 = sa * x - sm * pred_x0
            if i == len(t_levels) - 1:
                x = pred_x0
            else:
                t_prev = t_levels[i + 1].expand(B)
                abar_prev = diffusion._alpha_bar(t_prev).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
                noise = torch.randn_like(x) * (temperature if temperature != 1.0 else 1.0)
                x = abar_prev.sqrt() * pred_x0 + (1.0 - abar_prev).clamp(min=0).sqrt() * noise
        return x

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)

    def decode_embeddings_to_tokens(self, embeds: torch.Tensor) -> torch.Tensor:
        """Argmax over the LM head applied to denoised embeddings -> token ids."""
        logits = self.lm_head(embeds)
        return logits.argmax(dim=-1)

    def set_train_ctx(self):
        self.base_model.train()
