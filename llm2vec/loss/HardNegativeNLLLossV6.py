# HardNegativeNLLLossV6
# HardNegativeNLLLossV5 plus an optional auxiliary pooling regularization term.

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .loss_utils import mismatched_sizes_all_gather


def cos_sim(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    a = F.normalize(a, p=2, dim=-1, eps=eps)
    b = F.normalize(b, p=2, dim=-1, eps=eps)
    return a @ b.t()


class HardNegativeNLLLoss(nn.Module):
    """
    HardNegativeNLLLossV6:
      - retains the V5 top-k shared SlerpMixCSE formulation
      - optionally adds an auxiliary pooling loss, e.g. structured self-attention
        diversity regularization, to avoid touching the main pipeline contract
    """

    supports_aux_loss = True

    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
        lam: float = 0.2,
        interpolation_mode: str = "slerp",
        normalize_mixed: bool = True,
        slerp_eps: float = 1e-6,
        shared_mix_topk: int = 4,
        aux_loss_weight: float = 0.0,
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
        if aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be >= 0.")
        self.scale = float(scale)
        self.similarity_fct = similarity_fct
        self.lam = float(lam)
        self.interpolation_mode = interpolation_mode
        self.normalize_mixed = bool(normalize_mixed)
        self.slerp_eps = float(slerp_eps)
        self.shared_mix_topk = int(shared_mix_topk)
        self.aux_loss_weight = float(aux_loss_weight)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def _mix_representations(self, pos: Tensor, hard_neg: Tensor) -> Tensor:
        if self.interpolation_mode == "lerp":
            mixed = self.lam * pos + (1.0 - self.lam) * hard_neg
            if self.normalize_mixed:
                mixed = F.normalize(mixed, p=2, dim=-1)
            return mixed

        pos_norm = F.normalize(pos, p=2, dim=-1)
        hard_norm = F.normalize(hard_neg, p=2, dim=-1)
        t = torch.full(
            (pos_norm.size(0), 1),
            1.0 - self.lam,
            dtype=pos_norm.dtype,
            device=pos_norm.device,
        )

        dot = (pos_norm * hard_norm).sum(dim=-1, keepdim=True).clamp(
            -1.0 + self.slerp_eps, 1.0 - self.slerp_eps
        )
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        slerp_mixed = (
            torch.sin((1.0 - t) * theta) / sin_theta * pos_norm
            + torch.sin(t * theta) / sin_theta * hard_norm
        )
        lerp_fallback = F.normalize(
            (1.0 - t) * pos_norm + t * hard_norm, p=2, dim=-1
        )
        use_lerp = sin_theta.abs() < self.slerp_eps
        mixed = torch.where(use_lerp, lerp_fallback, slerp_mixed)

        if self.normalize_mixed:
            mixed = F.normalize(mixed, p=2, dim=-1)
        return mixed

    def forward(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Optional[Tensor] = None,
        aux_loss: Optional[Tensor] = None,
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

        batch_size = full_q_reps.size(0)
        device = full_q_reps.device

        logits_pos = self.similarity_fct(full_q_reps, full_d_reps_pos) * self.scale

        has_neg = full_d_reps_neg is not None and full_d_reps_neg.size(0) > 0
        if has_neg:
            logits_neg = self.similarity_fct(full_q_reps, full_d_reps_neg) * self.scale

            with torch.no_grad():
                neg_scores = logits_neg / self.scale
                hard_idx = torch.argmax(neg_scores, dim=1)
                hard_neg = full_d_reps_neg[hard_idx]

            mixed = self._mix_representations(full_d_reps_pos, hard_neg).detach()
            logits_mix_all = self.similarity_fct(full_q_reps, mixed) * self.scale
            topk = min(self.shared_mix_topk, logits_mix_all.size(1))
            logits_mix, _ = torch.topk(logits_mix_all, k=topk, dim=1)
            logits = torch.cat([logits_pos, logits_neg, logits_mix], dim=1)
        else:
            logits = logits_pos

        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        retrieval_loss = self.cross_entropy_loss(logits, labels)

        if aux_loss is None or self.aux_loss_weight == 0.0:
            return retrieval_loss
        return retrieval_loss + self.aux_loss_weight * aux_loss
