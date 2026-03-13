# Difficulty-adaptive MixCSE + margin-aware mixed-negative penalty
# Self-contained: includes cos_sim and mismatched_sizes_all_gather
# Supports single-GPU and DDP (torch.distributed)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from typing import Optional

from .loss_utils import mismatched_sizes_all_gather


def cos_sim(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Cosine similarity matrix between a [M,D] and b [N,D] -> [M,N].
    Safe even if inputs are not normalized.
    """
    a = F.normalize(a, p=2, dim=-1, eps=eps)
    b = F.normalize(b, p=2, dim=-1, eps=eps)
    return a @ b.t()


class HardNegativeNLLLoss(nn.Module):
    """
    HardNegativeNLLLossV3:
      - in-batch positives as candidates (diagonal is positive)
      - optional explicit negatives (d_reps_neg)
      - hard-negative mining from d_reps_neg (per query)
      - sample-wise dynamic lam_i driven by current hard-vs-pos difficulty
      - mixed negative per query: mixed_i = lam_i * pos_i + (1-lam_i) * hardneg_i
      - standard CE over candidates + extra margin-aware penalty on mixed negatives

    Notes:
      - lam_i is computed without gradient to keep the control signal stable.
      - the mixed branch is reweighted by its own margin, rather than by sample-level focal CE.
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

    def _compute_dynamic_lam(self, s_pos: Tensor, s_hard: Tensor) -> Tensor:
        # Harder samples (hard negatives close to positives) get larger lam.
        hardness = torch.sigmoid(self.hardness_alpha * (s_hard - s_pos))
        lam = self.lam_min + (self.lam_max - self.lam_min) * hardness
        return lam

    def forward(
        self,
        q_reps: Tensor,  # [B, D]
        d_reps_pos: Tensor,  # [B, D]
        d_reps_neg: Optional[Tensor] = None,  # [N, D] or None
    ) -> Tensor:
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if dist.is_available() and dist.is_initialized():
            full_q_reps = torch.cat(mismatched_sizes_all_gather(q_reps), dim=0)
            full_d_reps_pos = torch.cat(mismatched_sizes_all_gather(d_reps_pos), dim=0)
            full_d_reps_neg = torch.cat(mismatched_sizes_all_gather(d_reps_neg), dim=0)
        else:
            full_q_reps = q_reps
            full_d_reps_pos = d_reps_pos
            full_d_reps_neg = d_reps_neg

        if full_q_reps.size(0) != full_d_reps_pos.size(0):
            raise RuntimeError(
                f"Alignment error: full_q_reps has {full_q_reps.size(0)} rows, "
                f"but full_d_reps_pos has {full_d_reps_pos.size(0)} rows."
            )

        B = full_q_reps.size(0)
        device = full_q_reps.device

        logits_pos = self.similarity_fct(full_q_reps, full_d_reps_pos) * self.scale
        labels = torch.arange(B, device=device, dtype=torch.long)

        has_neg = full_d_reps_neg is not None and full_d_reps_neg.size(0) > 0
        if not has_neg:
            return self.cross_entropy_loss(logits_pos, labels)

        neg_scores = self.similarity_fct(full_q_reps, full_d_reps_neg)
        logits_neg = neg_scores * self.scale

        q_norm = F.normalize(full_q_reps, p=2, dim=-1)
        pos_norm = F.normalize(full_d_reps_pos, p=2, dim=-1)
        s_pos = (q_norm * pos_norm).sum(dim=-1)

        with torch.no_grad():
            hard_idx = torch.argmax(neg_scores, dim=1)
            s_hard = neg_scores.gather(1, hard_idx.unsqueeze(1)).squeeze(1)
            hard_neg = full_d_reps_neg[hard_idx]
            lam = self._compute_dynamic_lam(s_pos, s_hard).unsqueeze(1)

        mixed = lam * full_d_reps_pos + (1.0 - lam) * hard_neg
        if self.normalize_mixed:
            mixed = F.normalize(mixed, p=2, dim=-1)
        if self.detach_mixed:
            mixed = mixed.detach()

        s_mix = (q_norm * mixed).sum(dim=-1)
        logits_mix = (s_mix * self.scale).unsqueeze(1)

        logits = torch.cat([logits_pos, logits_neg, logits_mix], dim=1)
        base_loss = self.cross_entropy_loss(logits, labels)

        m_mix = s_pos - s_mix
        mix_difficulty = F.relu(self.margin_target - m_mix)
        if self.mix_gamma > 0.0:
            mix_weights = (1.0 + mix_difficulty).pow(self.mix_gamma)
        else:
            mix_weights = torch.ones_like(mix_difficulty)

        mix_penalty = F.softplus(self.scale * (s_mix - s_pos + self.margin_delta))
        mix_loss = (mix_weights * mix_penalty).mean()

        return base_loss + self.mix_weight * mix_loss


if __name__ == "__main__":
    torch.manual_seed(0)

    B, D, N = 8, 128, 32
    q = torch.randn(B, D)
    pos = torch.randn(B, D)
    neg = torch.randn(N, D)

    loss_fn = HardNegativeNLLLoss(
        scale=20.0,
        lam_min=0.1,
        lam_max=0.6,
        hardness_alpha=5.0,
        mix_weight=1.0,
        margin_target=0.15,
        mix_gamma=2.0,
    )
    loss = loss_fn(q, pos, neg)
    print("loss:", float(loss))
