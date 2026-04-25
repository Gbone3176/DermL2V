# HardNegativeNLLLossV3_2
# Difficulty-adaptive MixCSE rebuilt on V0_2 row-aligned negatives.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .HardNegativeNLLLossV0_2Common import (
    cos_sim,
    gather_v0_2_reps,
    mix_lerp,
    row_mixed_logits,
    row_negative_logits,
    row_similarity,
)


class HardNegativeNLLLoss(nn.Module):
    """
    Dynamic-lambda MixCSE on top of V0_2.

    Difficulty is computed from the aligned pair scores s(q_i, pos_i) and
    s(q_i, neg_i); no cross-row explicit negative mining is performed.
    """

    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
        lam_min: float = 0.1,
        lam_max: float = 0.6,
        hardness_alpha: float = 5.0,
        normalize_mixed: bool = True,
        detach_mixed: bool = True,
        mix_weight: float = 1.0,
        margin_target: float = 0.15,
        margin_delta: float = 0.0,
        mix_gamma: float = 2.0,
    ):
        super().__init__()
        if not (0.0 <= lam_min <= 1.0):
            raise ValueError("lam_min must be in [0,1].")
        if not (0.0 <= lam_max <= 1.0):
            raise ValueError("lam_max must be in [0,1].")
        if lam_min > lam_max:
            raise ValueError("lam_min must be <= lam_max.")
        if hardness_alpha <= 0.0:
            raise ValueError("hardness_alpha must be > 0.")
        if mix_weight < 0.0:
            raise ValueError("mix_weight must be >= 0.")
        if mix_gamma < 0.0:
            raise ValueError("mix_gamma must be >= 0.")
        self.scale = float(scale)
        self.similarity_fct = similarity_fct
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.hardness_alpha = float(hardness_alpha)
        self.normalize_mixed = bool(normalize_mixed)
        self.detach_mixed = bool(detach_mixed)
        self.mix_weight = float(mix_weight)
        self.margin_target = float(margin_target)
        self.margin_delta = float(margin_delta)
        self.mix_gamma = float(mix_gamma)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def _compute_dynamic_lam(self, s_pos: Tensor, s_neg: Tensor) -> Tensor:
        hardness = torch.sigmoid(self.hardness_alpha * (s_neg - s_pos))
        return self.lam_min + (self.lam_max - self.lam_min) * hardness

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
        with torch.no_grad():
            s_pos = row_similarity(self.similarity_fct, full_q_reps, full_d_reps_pos)
            s_neg = row_similarity(self.similarity_fct, full_q_reps, full_d_reps_neg)
            lam = self._compute_dynamic_lam(s_pos, s_neg).unsqueeze(1)

        mixed = mix_lerp(
            full_d_reps_pos,
            full_d_reps_neg,
            lam,
            self.normalize_mixed,
        )
        if self.detach_mixed:
            mixed = mixed.detach()

        s_pos = row_similarity(self.similarity_fct, full_q_reps, full_d_reps_pos)
        s_mix = row_similarity(self.similarity_fct, full_q_reps, mixed)
        logits_mix = row_mixed_logits(
            self.similarity_fct, full_q_reps, mixed, self.scale, same_as_pos
        )
        logits = torch.cat([logits_pos, logits_neg, logits_mix], dim=1)
        base_loss = self.cross_entropy_loss(logits, labels)

        m_mix = s_pos - s_mix
        mix_difficulty = F.relu(self.margin_target - m_mix)
        if same_as_pos.any():
            mix_difficulty = mix_difficulty.masked_fill(same_as_pos, 0.0)
        if self.mix_gamma > 0.0:
            mix_weights = (1.0 + mix_difficulty).pow(self.mix_gamma)
        else:
            mix_weights = torch.ones_like(mix_difficulty)
        mix_penalty = F.softplus(self.scale * (s_mix - s_pos + self.margin_delta))
        if same_as_pos.any():
            mix_penalty = mix_penalty.masked_fill(same_as_pos, 0.0)
        return base_loss + self.mix_weight * (mix_weights * mix_penalty).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(8, 128)
    pos = torch.randn(8, 128)
    neg = torch.randn(8, 128)
    print(float(HardNegativeNLLLoss()(q, pos, neg)))
