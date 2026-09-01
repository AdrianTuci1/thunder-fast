"""Continuous embedding-space diffusion with a masking component.

This is the training-side core of the DiffuLLaMA-style adaptation: an autoregressive
LLM (causal, next-token) is turned into a denoising model that reconstructs clean
embeddings from noisy/masked embeddings, given the diffusion timestep.

The parameterization follows the standard variance-preserving DDPM formulation in
continuous time (cosine alpha_bar), with an additional absorbing "mask" branch so a
fraction of positions are driven toward a learned mask embedding (hybrid continuous +
masking, as used in the DiffuLLaMA continuous-embedding variant).

We operate in the model's embedding space (before the LM head). The model predicts
`x0` (clean embedding), `epsilon` (noise) or `v` (velocity); the default is `x0`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def cosine_alpha_bar(t: Tensor, s: float = 0.008) -> Tensor:
    """alpha_bar(t) for the cosine noise schedule, t in [0, 1]."""
    return (torch.cos((t + s) / (1.0 + s) * math.pi / 2.0).clamp(min=1e-8)) ** 2


def linear_beta_schedule(beta_start: float, beta_end: float, steps: int) -> Tensor:
    return torch.linspace(beta_start, beta_end, steps)


class ContinuousDiffusion:
    def __init__(
        self,
        hidden_size: int,
        schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        train_steps: int = 24,
        infer_steps: int = 10,
        prediction: str = "x0",
        mask_ratio: float = 0.25,
    ):
        self.schedule = schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.train_steps = train_steps
        self.infer_steps = infer_steps
        self.prediction = prediction  # "x0" | "epsilon" | "v"
        self.mask_ratio = mask_ratio

        # Discrete betas for the training-time noise levels (for step-indexed schedules).
        if schedule == "linear":
            self.betas = linear_beta_schedule(beta_start, beta_end, train_steps)
        else:
            raise ValueError(f"Unsupported schedule: {schedule}")
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # A learned mask embedding is registered lazily by the model (needs hidden_size).
        self.hidden_size = hidden_size

    # ---- schedule helpers (continuous t in [0, 1]) ----
    def _alpha_bar(self, t: Tensor) -> Tensor:
        if self.schedule == "linear":
            # Interpolate between alpha_bar(0)=1 and alpha_bar(1)=alphas_cumprod[-1].
            start = torch.ones_like(t)
            end = self.alphas_cumprod[-1].to(t.device)
            return start + t * (end - start)
        return cosine_alpha_bar(t)

    @torch.no_grad()
    def _alpha_bar_klass(self, t: Tensor) -> Tensor:  # placeholder for discrete indexing
        return self._alpha_bar(t)

    # ---- forward (noising) ----
    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Add Gaussian noise to clean embeddings x0 at (continuous) timestep t."""
        abar = self._alpha_bar(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        sqrt_abar = abar.sqrt()
        sqrt_one_minus_abar = (1.0 - abar).sqrt()
        return sqrt_abar * x0 + sqrt_one_minus_abar * noise

    def _prediction_target(self, x0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        abar = self._alpha_bar(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        sqrt_abar = abar.sqrt()
        sqrt_one_minus_abar = (1.0 - abar).sqrt()
        if self.prediction == "x0":
            return x0
        if self.prediction == "epsilon":
            return eps
        if self.prediction == "v":
            return sqrt_abar * eps - sqrt_one_minus_abar * x0
        raise ValueError(self.prediction)

    def training_loss(
        self,
        model,
        x0: Tensor,           # [B, L, D] clean token embeddings
        mask_embedding: Tensor,  # [D] learned mask embedding
        t: Tensor,            # [B] float timesteps in [0, 1]
    ) -> Tensor:
        """One training batch: no use x0, sample timestep/emphasis, compute denoising loss."""
        B, L, D = x0.shape
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # Masking branch: replace a subset of positions with mask_embedding.
        use_mask = torch.rand(B, L, device=x0.device) < self.mask_ratio  # [B, L]
        mask_embedding = mask_embedding.view(1, 1, D).expand(B, L, D).clone()
        x_t = torch.where(use_mask.unsqueeze(-1), mask_embedding, x_t)
        # Masked positions get a higher loss weight so the model learns to reconstruct them.
        weight = torch.where(use_mask, torch.tensor(2.0, device=x0.device),
                             torch.tensor(1.0, device=x0.device))  # [B, L]
        weight = weight.unsqueeze(-1)

        pred = model(x_t, t)  # [B, L, D] predicted x0/eps/v (before LM head)

        target = self._prediction_target(x0, noise, t)
        loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1)  # [B, L]
        loss = (loss * weight.squeeze(-1)).sum() / (weight.sum() + 1e-8)
        return loss

    def inference_schedule(self, device: torch.device) -> Tensor:
        """Timesteps (ascending) used at inference for reverse denoising."""
        steps = self.infer_steps
        # Reverse diffusion goes from t=1 (pure noise) to t=0 (clean).
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)[:steps]
        return ts
