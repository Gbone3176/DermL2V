"""Losses and distributed helpers for contrastive embedding training."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()
        ops = [dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())]
        for op in ops:
            op.wait()
        return None, grad_list[rank], None, None


all_gather_with_grad = AllGatherWithGrad.apply


def mismatched_sizes_all_gather(tensor: Tensor, mismatched_axis: int = 0) -> list[Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return [tensor]

    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.shape[mismatched_axis]], dtype=torch.int64, device=tensor.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes_list = torch.cat(sizes).cpu().tolist()
    max_size = max(sizes_list)

    padded_shape = (
        *tensor.shape[:mismatched_axis],
        max_size,
        *tensor.shape[mismatched_axis + 1 :],
    )
    padded = torch.zeros(padded_shape, device=tensor.device, dtype=tensor.dtype)
    padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis]).copy_(tensor)

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    all_gather_with_grad(gathered, padded, None, False)
    return [
        gathered[rank].narrow(mismatched_axis, 0, sizes_list[rank])
        for rank in range(world_size)
    ]


def cos_sim(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    a = F.normalize(a, p=2, dim=-1, eps=eps)
    b = F.normalize(b, p=2, dim=-1, eps=eps)
    return a @ b.t()


class RowAlignedHardNegativeNLLLoss(nn.Module):
    """In-batch positive NLL with one row-aligned hard negative per query."""

    def __init__(self, scale: float = 50.0):
        super().__init__()
        self.scale = float(scale)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query: Tensor, positive: Tensor, negative: Tensor | None = None) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            query = torch.cat(mismatched_sizes_all_gather(query), dim=0)
            positive = torch.cat(mismatched_sizes_all_gather(positive), dim=0)
            if negative is not None:
                negative = torch.cat(mismatched_sizes_all_gather(negative), dim=0)

        if query.size(0) != positive.size(0):
            raise ValueError(
                f"query and positive must align row-wise, got {query.size(0)} and {positive.size(0)}"
            )

        logits_pos = cos_sim(query, positive) * self.scale
        logits = logits_pos

        if negative is not None and negative.size(0) > 0:
            if negative.size(0) != query.size(0):
                raise ValueError(
                    f"negative must align row-wise with query, got {negative.size(0)} and {query.size(0)}"
                )
            query_norm = F.normalize(query, p=2, dim=-1)
            negative_norm = F.normalize(negative, p=2, dim=-1)
            logits_neg = ((query_norm * negative_norm).sum(dim=-1) * self.scale).unsqueeze(1)

            same_as_positive = torch.norm(negative - positive, dim=-1) < 1e-6
            if same_as_positive.any():
                logits_neg = logits_neg.masked_fill(same_as_positive.unsqueeze(1), -1e9)

            logits = torch.cat([logits_pos, logits_neg], dim=1)

        labels = torch.arange(query.size(0), dtype=torch.long, device=query.device)
        return self.cross_entropy(logits, labels)
