"""Discrete masked diffusion language model (MDM) objective + inference schedule.

This replaces the earlier continuous embedding-space diffusion (ADR 0008/0009). An
autoregressive LLM (causal, next-token) is turned into a bidirectional denoiser that
reconstructs *token ids* from a partially masked sequence. The mask is a discrete
`[MASK]` token in the vocabulary (not a learned embedding vector), and the loss is
masked cross-entropy, which avoids the posterior collapse of continuous Gaussian
diffusion.

The objective (MDM = Masked Diffusion Model) uses per-sample mask ratios sampled
uniformly from [0, 1], reconstructs masked positions with cross-entropy, and adds a
"path" reweighting term derived from the model's own confidence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Padding / "ignore" tokens are handled separately; packed sequences have no padding.
IGNORE_INDEX = -100


class MaskedDiffusion:
    """Discrete masked diffusion objective.

    `training_loss` masks a random per-sample fraction of positions with the mask token,
    runs the model (bidirectional attention) and returns `mdm_loss + path_loss` computed
    only over masked positions. Mask ratios are sampled uniformly from a curriculum in
    `[mask_ratio_min, mask_ratio_max]`.
    """

    def __init__(
        self,
        infer_steps: int = 24,
        mask_ratio_min: float = 1 / 500,
        mask_ratio_max: float = 1 - 1 / 500,
    ):
        self.infer_steps = infer_steps
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max

    def _sample_mask_ratio(self, B: int, device: torch.device) -> Tensor:
        """Per-sample mask ratio from a uniform curriculum in [min, max]."""
        lo, hi = self.mask_ratio_min, self.mask_ratio_max
        if hi <= lo:
            return torch.full((B,), lo, dtype=torch.float, device=device)
        return torch.rand(B, device=device) * (hi - lo) + lo

    def training_loss(self, model, input_ids: Tensor) -> Tensor:
        """Masked cross-entropy loss for discrete MDM.

        `input_ids`: [B, L] clean token ids (packed, no padding). A per-batch fraction of
        positions is replaced with the mask token; the model runs bidirectionally and
        predicts logits. Only masked positions contribute to the loss, weighted by both a
        plain cross-entropy term and a confidence-scaled "path" term.
        """
        B, L = input_ids.shape
        r = self._sample_mask_ratio(B, input_ids.device)  # [B] per-sample ratio
        mask_token_id = model.mask_token_id
        use_mask = torch.rand(B, L, device=input_ids.device) < r.unsqueeze(-1)  # [B, L]
        x_m = torch.where(use_mask, torch.full_like(input_ids, mask_token_id), input_ids)

        logits = model(x_m)  # [B, L, V], bidirectional attention

        # Next-token alignment (matches the AR base and the reference): position i predicts
        # token i+1, so we drop the last logit and shift labels right by one.
        pred_logits = logits[..., :-1, :].contiguous()  # [B, L-1, V]
        labels = input_ids[..., 1:].contiguous()        # [B, L-1]
        valid = use_mask[..., 1:]                        # [B, L-1] only masked positions count

        token_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )  # [B*(L-1)]
        loss_mask = valid.reshape(-1).float()
        denom = loss_mask.sum() + 1e-8

        mdm_loss = (token_loss * loss_mask).sum() / denom

        # "path" reweighting: weight each token's CE by the model's own confidence (exp of
        # -CE, i.e. prob of the true token) scaled by 1/mask_ratio.
        r_exp = r.unsqueeze(-1).expand(B, L - 1).reshape(-1)
        path = (
            (-token_loss).exp().detach()
            * token_loss
            * (1.0 / r_exp.clamp(min=1e-8))
            * loss_mask
        ).sum() / denom

        return mdm_loss + path
