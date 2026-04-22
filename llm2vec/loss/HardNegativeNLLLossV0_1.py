import torch
from torch import nn, Tensor
from .loss_utils import cos_sim, mismatched_sizes_all_gather


class HardNegativeNLLLoss:
    # V0_1 的修改目的：
    # 相比 V0 直接把所有显式 negative 都当真负例，这一版先检查
    # “某个 negative 是否其实和某个 positive 是同一个向量”。
    # 如果是，就把对应 logit mask 掉，避免最明显的 lexical / duplicate false negative。
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

        d_reps = torch.cat([full_d_reps_pos, full_d_reps_neg], dim=0)
        scores = self.similarity_fct(full_q_reps, d_reps) * self.scale

        num_q = full_q_reps.size(0)
        num_pos = full_d_reps_pos.size(0)

        if full_d_reps_neg.size(0) > 0:
            # 这里做的是“完全相同向量级别”的 false negative 过滤，
            # 不是更宽泛的语义级 false negative 处理。
            # [num_neg, num_pos]: neg_k 是否与 pos_j 向量相同
            pos_rep = full_d_reps_pos.unsqueeze(0)   # [1, num_pos, dim]
            neg_rep = full_d_reps_neg.unsqueeze(1)   # [num_neg, 1, dim]
            is_false_neg = (torch.norm(neg_rep - pos_rep, dim=-1) < 1e-6)  # [num_neg, num_pos]

            # fn_mask[j, k] = True 表示对 query_j 而言 neg_k 是 false negative
            fn_mask = is_false_neg.T  # [num_pos, num_neg] == [num_q, num_neg]

            # V0_1 的做法是直接从 softmax 候选中删掉这些冲突 negative，
            # 即：既不把它当正例，也不继续把它当负例。
            full_mask = torch.zeros_like(scores, dtype=torch.bool)
            full_mask[:, num_pos:] = fn_mask
            scores = scores.masked_fill(full_mask, -1e9)

        labels = torch.arange(num_q, dtype=torch.long, device=scores.device)
        loss = self.cross_entropy_loss(scores, labels)
        return loss
