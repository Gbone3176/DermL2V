import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather


class HardNegativeNLLLoss:
    # V0_2 的修改目的：
    # 不再把显式 negative 放进“全局共享候选池”，
    # 而是要求 neg 与 query 按行对齐，让 neg_i 只参与 query_i 自己那一行的 loss。
    # 这样可以避免“某个样本本来只是 A 的 hard negative，却被错误地拿去当 B/C/... 的负例”。
    def __init__(
        self,
        scale: float = 50.0,
        similarity_fct=cos_sim,
    ):
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

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

        num_q = full_q_reps.size(0)
        num_pos = full_d_reps_pos.size(0)

        if num_q != num_pos:
            raise ValueError(
                f"q_reps and d_reps_pos must be aligned row-wise, got {num_q} and {num_pos}."
            )

        # 正样本部分仍然保留 in-batch 对比：每个 query 都和所有 positive 比，
        # 正确标签仍然是对角线位置。
        logits_pos = self.similarity_fct(full_q_reps, full_d_reps_pos) * self.scale
        logits = logits_pos

        if full_d_reps_neg.size(0) > 0:
            if full_d_reps_neg.size(0) != num_q:
                raise ValueError(
                    "d_reps_neg must be row-aligned with q_reps when using V0_2, "
                    f"got {full_d_reps_neg.size(0)} negatives for {num_q} queries."
                )

            # 这里不再构造 [B, N] 的共享 negative logits，
            # 而是只计算每个 query 与自己 neg 的点积，得到 [B, 1]。
            q_norm = F.normalize(full_q_reps, p=2, dim=-1)
            neg_norm = F.normalize(full_d_reps_neg, p=2, dim=-1)
            logits_neg = ((q_norm * neg_norm).sum(dim=-1) * self.scale).unsqueeze(1)

            # 如果某个专属 negative 实际上和自己的 positive 完全一样，
            # 仍然把它从该行里屏蔽，避免自相矛盾的监督信号。
            same_as_pos = torch.norm(full_d_reps_neg - full_d_reps_pos, dim=-1) < 1e-6
            if same_as_pos.any():
                logits_neg = logits_neg.masked_fill(same_as_pos.unsqueeze(1), -1e9)

            logits = torch.cat([logits_pos, logits_neg], dim=1)

        labels = torch.arange(num_q, dtype=torch.long, device=logits.device)
        loss = self.cross_entropy_loss(logits, labels)
        return loss
