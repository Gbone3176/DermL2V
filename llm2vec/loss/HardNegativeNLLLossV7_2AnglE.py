# HardNegativeNLLLossV7_2AnglE
# AnglE similarity variant rebuilt on V0_2 row-aligned raw negatives.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .HardNegativeNLLLossV0_2Common import (
    cos_sim,
    gather_v0_2_reps,
    mix_lerp_or_slerp,
    row_negative_logits,
)


def angle_sim(
    a: Tensor,
    b: Tensor,
    eps: float = 1e-12,
    pooling_strategy: str = "sum",
    pad_odd_dim: bool = True,
) -> Tensor:
    if a.size(-1) != b.size(-1):
        raise ValueError(
            f"Expected matching embedding dims, got {a.size(-1)} and {b.size(-1)}."
        )
    if a.size(-1) % 2 != 0:
        if not pad_odd_dim:
            raise ValueError(
                "AnglE-style similarity requires an even embedding dimension. "
                "Set pad_odd_dim=True to append one zero channel."
            )
        a = F.pad(a, (0, 1))
        b = F.pad(b, (0, 1))

    a_re, a_im = torch.chunk(a, 2, dim=-1)
    b_re, b_im = torch.chunk(b, 2, dim=-1)
    a_norm = torch.sqrt(
        torch.clamp((a_re.square() + a_im.square()).sum(dim=-1), min=eps)
    )
    b_norm = torch.sqrt(
        torch.clamp((b_re.square() + b_im.square()).sum(dim=-1), min=eps)
    )
    denom = torch.clamp(a_norm[:, None] * b_norm[None, :], min=eps)
    re = (a_re @ b_re.t() + a_im @ b_im.t()) / denom
    im = (a_im @ b_re.t() - a_re @ b_im.t()) / denom
    pooled = re + im
    if pooling_strategy == "mean":
        pooled = pooled / float(a.size(-1))
    elif pooling_strategy != "sum":
        raise ValueError(
            f"Unsupported angle pooling strategy: {pooling_strategy}. "
            "Use 'sum' or 'mean'."
        )
    return pooled.abs()


class HardNegativeNLLLoss(nn.Module):
    supports_aux_loss = True

    def __init__(
        self,
        scale: float = 20.0,
        lam: float = 0.2,
        interpolation_mode: str = "slerp",
        normalize_mixed: bool = True,
        slerp_eps: float = 1e-6,
        shared_mix_topk: int = 4,
        aux_loss_weight: float = 0.0,
        cosine_weight: float = 0.0,
        angle_weight: float = 1.0,
        angle_pooling_strategy: str = "sum",
        angle_pad_odd_dim: bool = True,
        similarity_eps: float = 1e-12,
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
        if cosine_weight < 0.0 or angle_weight < 0.0:
            raise ValueError("cosine_weight and angle_weight must be >= 0.")
        if cosine_weight == 0.0 and angle_weight == 0.0:
            raise ValueError("At least one of cosine_weight or angle_weight must be > 0.")
        if angle_pooling_strategy not in {"sum", "mean"}:
            raise ValueError("angle_pooling_strategy must be 'sum' or 'mean'.")
        if similarity_eps <= 0.0:
            raise ValueError("similarity_eps must be > 0.")
        self.scale = float(scale)
        self.lam = float(lam)
        self.interpolation_mode = interpolation_mode
        self.normalize_mixed = bool(normalize_mixed)
        self.slerp_eps = float(slerp_eps)
        self.shared_mix_topk = int(shared_mix_topk)
        self.aux_loss_weight = float(aux_loss_weight)
        self.cosine_weight = float(cosine_weight)
        self.angle_weight = float(angle_weight)
        self.angle_pooling_strategy = angle_pooling_strategy
        self.angle_pad_odd_dim = bool(angle_pad_odd_dim)
        self.similarity_eps = float(similarity_eps)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def _similarity(self, a: Tensor, b: Tensor) -> Tensor:
        sim = None
        if self.cosine_weight > 0.0:
            sim = self.cosine_weight * cos_sim(a, b, eps=self.similarity_eps)
        if self.angle_weight > 0.0:
            angle_component = self.angle_weight * angle_sim(
                a,
                b,
                eps=self.similarity_eps,
                pooling_strategy=self.angle_pooling_strategy,
                pad_odd_dim=self.angle_pad_odd_dim,
            )
            sim = angle_component if sim is None else sim + angle_component
        return sim

    def forward(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Optional[Tensor] = None,
        aux_loss: Optional[Tensor] = None,
    ) -> Tensor:
        full_q_reps, full_d_reps_pos, full_d_reps_neg = gather_v0_2_reps(
            q_reps, d_reps_pos, d_reps_neg
        )
        batch_size = full_q_reps.size(0)
        labels = torch.arange(batch_size, device=full_q_reps.device, dtype=torch.long)
        logits_pos = self._similarity(full_q_reps, full_d_reps_pos) * self.scale

        if full_d_reps_neg.size(0) > 0:
            logits_neg, same_as_pos = row_negative_logits(
                self._similarity,
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
            logits_mix_all = self._similarity(full_q_reps, mixed) * self.scale
            if same_as_pos.any():
                logits_mix_all = logits_mix_all.masked_fill(
                    same_as_pos.unsqueeze(0), -1e9
                )
            topk = min(self.shared_mix_topk, logits_mix_all.size(1))
            logits_mix, _ = torch.topk(logits_mix_all, k=topk, dim=1)
            logits = torch.cat([logits_pos, logits_neg, logits_mix], dim=1)
        else:
            logits = logits_pos

        retrieval_loss = self.cross_entropy_loss(logits, labels)
        if aux_loss is None or self.aux_loss_weight == 0.0:
            return retrieval_loss
        return retrieval_loss + self.aux_loss_weight * aux_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(8, 128)
    pos = torch.randn(8, 128)
    neg = torch.randn(8, 128)
    print(float(HardNegativeNLLLoss()(q, pos, neg)))
