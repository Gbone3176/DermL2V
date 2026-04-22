import torch
import torch.nn.functional as F
from torch import Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather


class HardNegativeNLLLoss:
    # V0_3 的修改目的：
    # 和 V0_1 一样，先检测“negative 是否其实与某个 positive 完全相同”；
    # 但处理方式不再是 mask 掉，而是把这些 false negatives 并入正类集合，
    # 用 multi-positive 的方式计算 loss。
    # 直觉上，相当于承认这些列本质上和主 positive 表达的是同一个目标。
    def __init__(
        self,
        scale: float = 50.0,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct

    def __call__(
        self,
        q_reps: Tensor,
        d_reps_pos: Tensor,
        d_reps_neg: Tensor = None,
    ):
        if d_reps_neg is None:
            d_reps_neg = d_reps_pos[:0, :]

        if torch.distributed.is_initialized():
            full_d_reps_pos = torch.cat(mismatched_sizes_all_gather(d_reps_pos))
            full_q_reps = torch.cat(mismatched_sizes_all_gather(q_reps))
            full_d_reps_neg = torch.cat(mismatched_sizes_all_gather(d_reps_neg))
        else:
            full_d_reps_pos = d_reps_pos
            full_q_reps = q_reps
            full_d_reps_neg = d_reps_neg

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self.similarity_fct(full_q_reps, d_reps) * self.scale

        num_q = full_q_reps.size(0)
        num_pos = full_d_reps_pos.size(0)
        num_neg = full_d_reps_neg.size(0)

        if num_q != num_pos:
            raise ValueError(
                f"q_reps and d_reps_pos must align row-wise, got {num_q} and {num_pos}."
            )

        # 先把标准对角线 positive 标出来。
        positive_mask = torch.zeros_like(scores, dtype=torch.bool)
        positive_mask[torch.arange(num_q, device=scores.device), torch.arange(num_pos, device=scores.device)] = True

        if num_neg > 0:
            pos_rep = full_d_reps_pos.unsqueeze(0)   # [1, num_pos, dim]
            neg_rep = full_d_reps_neg.unsqueeze(1)   # [num_neg, 1, dim]
            is_false_neg = torch.norm(neg_rep - pos_rep, dim=-1) < 1e-6  # [num_neg, num_pos]

            # 对 query_j 而言，任何与 pos_j 完全相同的 neg_k，
            # 都被视为“额外的正例列”，而不是被删掉。
            fn_mask = is_false_neg.T  # [num_pos, num_neg] == [num_q, num_neg]
            positive_mask[:, num_pos:] = fn_mask

        # 这里不再做普通 CE，而是做 multi-positive NLL：
        # 分子 = 所有正例列的 logsumexp
        # 分母 = 全部候选列的 logsumexp
        neg_inf = torch.finfo(scores.dtype).min
        positive_scores = scores.masked_fill(~positive_mask, neg_inf)

        log_denom = torch.logsumexp(scores, dim=1)
        log_numer = torch.logsumexp(positive_scores, dim=1)
        loss = -(log_numer - log_denom).mean()
        return loss
