# HardNegativeNLLLossV4
# MixCSE-style hard-negative NLL with fixed interpolation between the aligned. Slerp-based Mix 
# positive document and the mined hardest explicit negative.
# Supports single-GPU and DDP (torch.distributed).

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .loss_utils import mismatched_sizes_all_gather


def cos_sim(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Return the cosine-similarity matrix between `a` [M, D] and `b` [N, D].
    Inputs are normalized internally for numerical safety.
    """
    a = F.normalize(a, p=2, dim=-1, eps=eps)
    b = F.normalize(b, p=2, dim=-1, eps=eps)
    return a @ b.t()


class HardNegativeNLLLoss(nn.Module):
    """
    HardNegativeNLLLossV4:
      - uses in-batch aligned positives as the base candidate set
      - optionally adds an explicit negative pool `d_reps_neg`
      - mines the hardest explicit negative for each query
      - constructs one mixed negative per query from:
        positive_i and hardest_negative_i
      - appends that mixed negative as a row-specific extra logit
      - blocks gradients through the mixed negative branch via `detach()`

    Interpolation behavior:
      - `lerp`: linear interpolation in embedding space
      - `slerp`: spherical interpolation after L2 normalization

    Notes:
      - `lam` is a fixed constant, not a sampled or adaptive weight.
      - diagonal labels assume `q_reps[i]` aligns with `d_reps_pos[i]`.
      - mixed negatives are not shared across rows; each query gets exactly one.
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

    def _mix_representations(self, pos: Tensor, hard_neg: Tensor) -> Tensor:
        """
        Build one mixed negative per row from the aligned positive and the
        mined hardest negative.
        """
        if self.interpolation_mode == "lerp":
            mixed = self.lam * pos + (1.0 - self.lam) * hard_neg
            if self.normalize_mixed:
                mixed = F.normalize(mixed, p=2, dim=-1)
            return mixed

        # SLERP operates on normalized vectors so interpolation follows the
        # hypersphere instead of the raw Euclidean chord.
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
        # When the angle is too small, the SLERP denominator becomes unstable.
        use_lerp = sin_theta.abs() < self.slerp_eps
        mixed = torch.where(use_lerp, lerp_fallback, slerp_mixed)

        if self.normalize_mixed:
            mixed = F.normalize(mixed, p=2, dim=-1)
        return mixed

    def forward(
        self,
        q_reps: Tensor,  # [B, D]
        d_reps_pos: Tensor,  # [B, D], aligned one-to-one with q_reps
        d_reps_neg: Optional[Tensor] = None,  # [N, D] explicit negatives
    ) -> Tensor:
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        # Gather variable-sized local batches across ranks when running DDP.
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

        # Base candidate matrix: each row predicts its aligned positive on the diagonal.
        logits_pos = (
            self.similarity_fct(full_q_reps, full_d_reps_pos) * self.scale
        )

        has_neg = full_d_reps_neg is not None and full_d_reps_neg.size(0) > 0
        if has_neg:
            logits_neg = (
                self.similarity_fct(full_q_reps, full_d_reps_neg) * self.scale
            )

            # Select the most similar explicit negative for each query.
            with torch.no_grad():
                neg_scores = logits_neg / self.scale
                hard_idx = torch.argmax(neg_scores, dim=1)
                hard_neg = full_d_reps_neg[hard_idx]

            # Mixed negatives behave as harder row-specific distractors, but the
            # branch is detached so it does not receive direct gradient updates.
            mixed = self._mix_representations(full_d_reps_pos, hard_neg).detach()

            q_norm = F.normalize(full_q_reps, p=2, dim=-1)
            m_norm = mixed if self.normalize_mixed else F.normalize(mixed, p=2, dim=-1)
            logits_mix = ((q_norm * m_norm).sum(dim=-1) * self.scale).unsqueeze(1)

            # Final layout per row: [all in-batch positives | all explicit negatives | own mixed negative]
            logits = torch.cat([logits_pos, logits_neg, logits_mix], dim=1)
        else:
            logits = logits_pos

        # Positive label is always the aligned document on the diagonal.
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        return self.cross_entropy_loss(logits, labels)


if __name__ == "__main__":
    torch.manual_seed(0)

    B, D, N = 8, 128, 32
    q = torch.randn(B, D)
    pos = torch.randn(B, D)
    neg = torch.randn(N, D)

    loss_fn = HardNegativeNLLLoss(
        scale=20.0,
        lam=0.2,
        interpolation_mode="slerp",
    )
    loss = loss_fn(q, pos, neg)
    print("loss:", float(loss))
