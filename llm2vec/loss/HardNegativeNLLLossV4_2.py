# HardNegativeNLLLossV4_2
# SlerpMixCSE rebuilt on V0_2 row-aligned explicit-negative semantics.

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .HardNegativeNLLLossV0_2Common import (
    cos_sim,
    gather_v0_2_reps,
    mix_lerp_or_slerp,
    row_mixed_logits,
    row_negative_logits,
)


class HardNegativeNLLLoss(nn.Module):
    """
    V4 interpolation choices on top of V0_2.

    The explicit negative is not mined from a shared pool. The row-aligned
    neg_i is used directly with pos_i to construct the row-specific mixed
    negative.
    """

    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
        lam: float = 0.2,
        interpolation_mode: str = "slerp",
        normalize_mixed: bool = True,
        slerp_eps: float = 1e-6,
    ):
        super().__init__()
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0,1].")
        if interpolation_mode not in {"lerp", "slerp"}:
            raise ValueError("interpolation_mode must be 'lerp' or 'slerp'.")
        if slerp_eps <= 0.0:
            raise ValueError("slerp_eps must be > 0.")
        self.scale = float(scale)
        self.similarity_fct = similarity_fct
        self.lam = float(lam)
        self.interpolation_mode = interpolation_mode
        self.normalize_mixed = bool(normalize_mixed)
        self.slerp_eps = float(slerp_eps)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

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
        logits_mix = row_mixed_logits(
            self.similarity_fct, full_q_reps, mixed, self.scale, same_as_pos
        )
        logits = torch.cat([logits_pos, logits_neg, logits_mix], dim=1)
        return self.cross_entropy_loss(logits, labels)


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(8, 128)
    pos = torch.randn(8, 128)
    neg = torch.randn(8, 128)
    print(float(HardNegativeNLLLoss()(q, pos, neg)))
