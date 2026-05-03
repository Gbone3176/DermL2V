"""Small encoder models for contrastive embedding fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BertEmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "mean",
        normalize: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            add_pooling_layer=False,
        )
        self.pooling = pooling
        self.normalize = normalize

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)

        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = mean_pool(outputs.last_hidden_state, attention_mask)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
