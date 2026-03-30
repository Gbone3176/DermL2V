import torch
from torch import nn, Tensor
from typing import Optional

from .loss_utils import cos_sim, mismatched_sizes_all_gather


class HardNegativeNLLLoss:
    """
    Original V0 hard-negative NLL plus an optional auxiliary loss term.

    This class is intentionally isolated for the structured self-attention
    ablation so the original HardNegativeNLLLossV0 implementation stays
    untouched.
    """

    supports_aux_loss = True

    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
        aux_loss_weight: float = 0.0,
    ):
        if aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be >= 0.")
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.aux_loss_weight = float(aux_loss_weight)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Optional[Tensor] = None,
        aux_loss: Optional[Tensor] = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = mismatched_sizes_all_gather(d_reps_pos)
            full_d_reps_pos = torch.cat(full_d_reps_pos)

            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)

            full_d_reps_neg = mismatched_sizes_all_gather(d_reps_neg)
            full_d_reps_neg = torch.cat(full_d_reps_neg)
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self.similarity_fct(full_q_reps, d_reps) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )

        retrieval_loss = self.cross_entropy_loss(scores, labels)
        if aux_loss is None or self.aux_loss_weight == 0.0:
            return retrieval_loss
        return retrieval_loss + self.aux_loss_weight * aux_loss
