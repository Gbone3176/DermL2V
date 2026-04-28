# HardNegativeNLLLossV5_2_2
# V5_2 with each row's own mixed negative forced into the mixed candidates.

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .HardNegativeNLLLossV0_2Common import (
    cos_sim,
    gather_v0_2_reps,
    mix_lerp_or_slerp,
    row_negative_logits,
)


class HardNegativeNLLLoss(nn.Module):
    """
    V5_2 variant that preserves the row-aligned MixCSE signal.

    Raw explicit negatives remain row-private. Mixed negatives are still shared,
    but every query always receives its own mixed candidate first, then the
    remaining mixed slots are filled from the hardest non-own shared candidates.
    """

    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
        lam: float = 0.2,
        interpolation_mode: str = "slerp",
        normalize_mixed: bool = True,
        slerp_eps: float = 1e-6,
        shared_mix_topk: int = 4,
    ):
        super().__init__()
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0,1].")
        if interpolation_mode not in {"lerp", "slerp"}:
            raise ValueError("interpolation_mode must be 'lerp' or 'slerp'.")
        if slerp_eps <= 0.0:
            raise ValueError("slerp_eps must be > 0.")
        if shared_mix_topk <= 0:
            raise ValueError("shared_mix_topk must be > 0.")
        self.scale = float(scale)
        self.similarity_fct = similarity_fct
        self.lam = float(lam)
        self.interpolation_mode = interpolation_mode
        self.normalize_mixed = bool(normalize_mixed)
        self.slerp_eps = float(slerp_eps)
        self.shared_mix_topk = int(shared_mix_topk)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def _own_plus_shared_mix_logits(
        self,
        logits_mix_all: Tensor,
    ) -> Tensor:
        batch_size = logits_mix_all.size(0)
        own_logits = logits_mix_all.diagonal().contiguous().unsqueeze(1)
        shared_k = min(self.shared_mix_topk - 1, max(batch_size - 1, 0))
        if shared_k == 0:
            return own_logits

        eye = torch.eye(batch_size, device=logits_mix_all.device, dtype=torch.bool)
        shared_logits_all = logits_mix_all.masked_fill(eye, -1e9)
        shared_logits, _ = torch.topk(shared_logits_all, k=shared_k, dim=1)
        return torch.cat([own_logits, shared_logits], dim=1)

    def forward(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Optional[Tensor] = None,
    ) -> Tensor:
        full_q_reps, full_d_reps_pos, full_d_reps_neg = gather_v0_2_reps(
            q_reps, d_reps_pos, d_reps_neg
        )
        batch_size = full_q_reps.size(0)
        labels = torch.arange(batch_size, device=full_q_reps.device, dtype=torch.long)
        logits_pos = self.similarity_fct(full_q_reps, full_d_reps_pos) * self.scale

        if full_d_reps_neg.size(0) == 0:
            return self.cross_entropy_loss(logits_pos, labels)

        logits_neg, same_as_pos = row_negative_logits(
            self.similarity_fct,
            full_q_reps,
            full_d_reps_pos,
            full_d_reps_neg,
            self.scale,
        )
        mixed = mix_lerp_or_slerp(
            full_d_reps_pos,
            full_d_reps_neg,
            self.lam,
            self.interpolation_mode,
            self.normalize_mixed,
            self.slerp_eps,
        ).detach()
        logits_mix_all = self.similarity_fct(full_q_reps, mixed) * self.scale
        if same_as_pos.any():
            logits_mix_all = logits_mix_all.masked_fill(same_as_pos.unsqueeze(0), -1e9)
        logits_mix = self._own_plus_shared_mix_logits(logits_mix_all)
        logits = torch.cat([logits_pos, logits_neg, logits_mix], dim=1)
        return self.cross_entropy_loss(logits, labels)


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(8, 128)
    pos = torch.randn(8, 128)
    neg = torch.randn(8, 128)
    print(float(HardNegativeNLLLoss()(q, pos, neg)))
