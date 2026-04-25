from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .loss_utils import mismatched_sizes_all_gather


def cos_sim(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    a = F.normalize(a, p=2, dim=-1, eps=eps)
    b = F.normalize(b, p=2, dim=-1, eps=eps)
    return a @ b.t()


def gather_v0_2_reps(
    q_reps: Tensor,
    d_reps_pos: Tensor,
    d_reps_neg: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
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

    if full_d_reps_neg.size(0) not in {0, full_q_reps.size(0)}:
        raise ValueError(
            "d_reps_neg must be empty or row-aligned with q_reps for V0_2-based "
            f"losses, got {full_d_reps_neg.size(0)} negatives for "
            f"{full_q_reps.size(0)} queries."
        )

    return full_q_reps, full_d_reps_pos, full_d_reps_neg


def row_similarity(
    similarity_fct: Callable[[Tensor, Tensor], Tensor],
    q_reps: Tensor,
    d_reps: Tensor,
) -> Tensor:
    return similarity_fct(q_reps, d_reps).diagonal().contiguous()


def row_negative_logits(
    similarity_fct: Callable[[Tensor, Tensor], Tensor],
    q_reps: Tensor,
    d_reps_pos: Tensor,
    d_reps_neg: Tensor,
    scale: float,
) -> Tuple[Tensor, Tensor]:
    neg_scores = row_similarity(similarity_fct, q_reps, d_reps_neg)
    logits_neg = (neg_scores * scale).unsqueeze(1)
    same_as_pos = torch.norm(d_reps_neg - d_reps_pos, dim=-1) < 1e-6
    if same_as_pos.any():
        logits_neg = logits_neg.masked_fill(same_as_pos.unsqueeze(1), -1e9)
    return logits_neg, same_as_pos


def row_mixed_logits(
    similarity_fct: Callable[[Tensor, Tensor], Tensor],
    q_reps: Tensor,
    mixed: Tensor,
    scale: float,
    invalid_rows: Optional[Tensor] = None,
) -> Tensor:
    logits_mix = (row_similarity(similarity_fct, q_reps, mixed) * scale).unsqueeze(1)
    if invalid_rows is not None and invalid_rows.any():
        logits_mix = logits_mix.masked_fill(invalid_rows.unsqueeze(1), -1e9)
    return logits_mix


def mix_lerp(
    pos: Tensor,
    neg: Tensor,
    lam: Union[float, Tensor],
    normalize_mixed: bool = True,
) -> Tensor:
    mixed = lam * pos + (1.0 - lam) * neg
    if normalize_mixed:
        mixed = F.normalize(mixed, p=2, dim=-1)
    return mixed


def mix_lerp_or_slerp(
    pos: Tensor,
    neg: Tensor,
    lam: float,
    interpolation_mode: str,
    normalize_mixed: bool = True,
    slerp_eps: float = 1e-6,
) -> Tensor:
    if interpolation_mode == "lerp":
        return mix_lerp(pos, neg, lam, normalize_mixed)

    pos_norm = F.normalize(pos, p=2, dim=-1)
    neg_norm = F.normalize(neg, p=2, dim=-1)
    t = torch.full(
        (pos_norm.size(0), 1),
        1.0 - lam,
        dtype=pos_norm.dtype,
        device=pos_norm.device,
    )

    dot = (pos_norm * neg_norm).sum(dim=-1, keepdim=True).clamp(
        -1.0 + slerp_eps, 1.0 - slerp_eps
    )
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    slerp_mixed = (
        torch.sin((1.0 - t) * theta) / sin_theta * pos_norm
        + torch.sin(t * theta) / sin_theta * neg_norm
    )
    lerp_fallback = F.normalize((1.0 - t) * pos_norm + t * neg_norm, p=2, dim=-1)
    mixed = torch.where(sin_theta.abs() < slerp_eps, lerp_fallback, slerp_mixed)
    if normalize_mixed:
        mixed = F.normalize(mixed, p=2, dim=-1)
    return mixed
