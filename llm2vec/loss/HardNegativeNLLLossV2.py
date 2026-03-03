# MixCSE (lam) + Focal Style reweighting (gamma) + DDP support
# Self-contained: includes cos_sim and mismatched_sizes_all_gather
# Supports single-GPU and DDP (torch.distributed)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from typing import List, Optional
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
    MixCSE-style loss with:
      - in-batch positives as candidates (diagonal is positive)
      - optional explicit negatives (d_reps_neg)
      - hard-negative mining from d_reps_neg (per query)
      - mixed negative per query: mixed_i = normalize(lam * pos_i + (1-lam) * hardneg_i)
      - stop-gradient on mixed negative (detach)

    IMPORTANT:
      - lam is FIXED constant (no random sampling).
      - mixed negatives are appended PER-ROW as an extra logit, not as a global shared candidate pool.
    """
    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
        lam: float = 0.2,
        normalize_mixed: bool = True,
        gamma: float = 0.5,
    ):
        super().__init__()
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0,1].")
        if gamma < 0:
            raise ValueError("gamma must be >= 0.")
        self.scale = float(scale)
        self.similarity_fct = similarity_fct
        self.lam = float(lam)
        self.normalize_mixed = bool(normalize_mixed)
        self.gamma = float(gamma)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        q_reps: Tensor,         # [B, D]
        d_reps_pos: Tensor,     # [B, D]  (aligned with q_reps)
        d_reps_neg: Optional[Tensor] = None,  # [N, D] or None
    ) -> Tensor:
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        # -------- DDP gather (handles different local batch sizes) --------
        if dist.is_available() and dist.is_initialized():
            full_q_reps = torch.cat(mismatched_sizes_all_gather(q_reps), dim=0)
            full_d_reps_pos = torch.cat(mismatched_sizes_all_gather(d_reps_pos), dim=0)
            full_d_reps_neg = torch.cat(mismatched_sizes_all_gather(d_reps_neg), dim=0)
        else:
            full_q_reps = q_reps
            full_d_reps_pos = d_reps_pos
            full_d_reps_neg = d_reps_neg

        # Safety: q and pos must align 1-to-1 for diagonal labels to be correct
        if full_q_reps.size(0) != full_d_reps_pos.size(0):
            raise RuntimeError(
                f"Alignment error: full_q_reps has {full_q_reps.size(0)} rows, "
                f"but full_d_reps_pos has {full_d_reps_pos.size(0)} rows."
            )

        B = full_q_reps.size(0)
        device = full_q_reps.device

        # -------- base logits: in-batch positives --------
        logits_pos = self.similarity_fct(full_q_reps, full_d_reps_pos) * self.scale  # [B, B]

        # -------- optional explicit negatives --------
        has_neg = full_d_reps_neg is not None and full_d_reps_neg.size(0) > 0
        if has_neg:
            logits_neg = self.similarity_fct(full_q_reps, full_d_reps_neg) * self.scale  # [B, N]

            # hard negative mining (no grad through selection)
            with torch.no_grad():
                neg_scores = logits_neg / self.scale  # [B, N]
                hard_idx = torch.argmax(neg_scores, dim=1)                      # [B]
                hard_neg = full_d_reps_neg[hard_idx]                            # [B, D]

            # mixed negative per query (stop-gradient)
            mixed = self.lam * full_d_reps_pos + (1.0 - self.lam) * hard_neg     # [B, D]
            if self.normalize_mixed:
                mixed = F.normalize(mixed, p=2, dim=-1)
            mixed = mixed.detach()

            # Only keep the query's own mixed negative instead of every cross-pair to avoid 3D logits.
            q_norm = F.normalize(full_q_reps, p=2, dim=-1)
            m_norm = mixed  # mixed 已 normalize
            logits_mix = ((q_norm * m_norm).sum(dim=-1) * self.scale).unsqueeze(1)  # [B,1]

            logits = torch.cat([logits_pos, logits_neg, logits_mix], dim=1)  # [B, B+N+1]
        else:
            logits = logits_pos  # [B, B]

        labels = torch.arange(B, device=device, dtype=torch.long)  # diagonal positives

        # -------- Focal-style reweighting (Eq. 4) --------
        # When gamma == 0, w_i = 1 and this reduces to standard cross-entropy.
        if self.gamma > 0.0:
            # log-probabilities and probabilities of the positive class
            log_probs = F.log_softmax(logits, dim=1)                          # [B, C]
            probs = torch.exp(log_probs)                                       # [B, C]
            # probability assigned to the true positive for each query
            pos_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)        # [B]
            # focal weight: harder samples (lower pos_prob) get higher weight
            focal_weights = (1.0 - pos_probs).pow(self.gamma)                 # [B]
            # weighted NLL: -w_i * log p(pos_i)
            pos_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
            loss = -(focal_weights * pos_log_probs).mean()
        else:
            loss = self.cross_entropy_loss(logits, labels)

        return loss


# ---------------------------
# Minimal sanity check usage
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, D, N = 8, 128, 32
    q = torch.randn(B, D)
    pos = torch.randn(B, D)
    neg = torch.randn(N, D)

    # gamma=0 -> standard CE
    loss_fn = HardNegativeNLLLoss(scale=20.0, lam=0.2, gamma=0.0)
    loss = loss_fn(q, pos, neg)
    print("loss (gamma=0):", float(loss))

    # gamma=2 -> focal reweighting (focus on hard samples)
    loss_fn_focal = HardNegativeNLLLoss(scale=20.0, lam=0.2, gamma=2.0)
    loss_focal = loss_fn_focal(q, pos, neg)
    print("loss (gamma=2):", float(loss_focal))
