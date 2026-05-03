import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    query_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query_embeds = F.normalize(query_embeds.float(), dim=-1)
    positive_embeds = F.normalize(positive_embeds.float(), dim=-1)
    negative_embeds = F.normalize(negative_embeds.float(), dim=-1)

    candidates = torch.cat([positive_embeds, negative_embeds], dim=0)
    logits_q = query_embeds @ candidates.t()
    logits_q = logits_q / temperature
    labels = torch.arange(query_embeds.size(0), device=query_embeds.device)
    loss_q = F.cross_entropy(logits_q, labels)

    logits_p = (positive_embeds @ query_embeds.t()) / temperature
    loss_p = F.cross_entropy(logits_p, labels)
    return 0.5 * (loss_q + loss_p)
