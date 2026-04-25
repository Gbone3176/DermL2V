# HardNegativeNLLLossV0_2StructuredSelfAttnAblation
# V0_2 retrieval objective plus optional structured self-attention aux loss.

from typing import Optional

import torch
from torch import Tensor, nn

from .HardNegativeNLLLossV0_2Common import (
    cos_sim,
    gather_v0_2_reps,
    row_negative_logits,
)


class HardNegativeNLLLoss:
    supports_aux_loss = True

    def __init__(
        self,
        scale: float = 20.0,
        similarity_fct=cos_sim,
        aux_loss_weight: float = 0.0,
    ):
        if aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be >= 0.")
        self.scale = float(scale)
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
        full_q_reps, full_d_reps_pos, full_d_reps_neg = gather_v0_2_reps(
            q_reps, d_reps_pos, d_reps_neg
        )
        batch_size = full_q_reps.size(0)
        labels = torch.arange(batch_size, device=full_q_reps.device, dtype=torch.long)
        logits_pos = self.similarity_fct(full_q_reps, full_d_reps_pos) * self.scale

        if full_d_reps_neg.size(0) > 0:
            logits_neg, _ = row_negative_logits(
                self.similarity_fct,
                full_q_reps,
                full_d_reps_pos,
                full_d_reps_neg,
                self.scale,
            )
            logits = torch.cat([logits_pos, logits_neg], dim=1)
        else:
            logits = logits_pos

        retrieval_loss = self.cross_entropy_loss(logits, labels)
        if aux_loss is None or self.aux_loss_weight == 0.0:
            return retrieval_loss
        return retrieval_loss + self.aux_loss_weight * aux_loss
