import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=self.eps)


class StructuredSelfAttentionPooling(nn.Module):
    """
    Multi-hop structured self-attention pooling inspired by
    "A Structured Self-attentive Sentence Embedding".

    Internally produces a structured sentence matrix M with shape (B, r, D)
    and then projects it back to a single-vector embedding of shape (B, D)
    for compatibility with the existing LLM2Vec training/eval pipeline.
    """

    def __init__(
        self,
        d_model: int,
        attn_hidden_dim: int = 512,
        num_hops: int = 8,
        output_dropout: float = 0.0,
        output_norm: str | None = "layernorm",
        gamma_init: float = 1e-2,
        gamma_learnable: bool = True,
        merge_mode: str = "residual",
        merge_temperature: float = 1.0,
        merge_hidden_dim: int | None = None,
        merge_input_norm: str | None = None,
        merge_mean_bias: float = 0.0,
        eps: float = 1e-9,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if attn_hidden_dim <= 0:
            raise ValueError("attn_hidden_dim must be positive.")
        if num_hops <= 0:
            raise ValueError("num_hops must be positive.")
        if gamma_init < 0:
            raise ValueError("gamma_init must be non-negative.")
        if output_norm not in {None, "none", "layernorm", "l2"}:
            raise ValueError("output_norm must be one of: None, 'none', 'layernorm', 'l2'.")
        if merge_mode not in {"residual", "router"}:
            raise ValueError("merge_mode must be one of: 'residual', 'router'.")
        if merge_temperature <= 0:
            raise ValueError("merge_temperature must be positive.")
        if merge_input_norm not in {None, "none", "layernorm", "l2"}:
            raise ValueError("merge_input_norm must be one of: None, 'none', 'layernorm', 'l2'.")

        self.d_model = int(d_model)
        self.attn_hidden_dim = int(attn_hidden_dim)
        self.num_hops = int(num_hops)
        self.output_dropout = float(output_dropout)
        self.output_norm_type = None if output_norm in {None, "none"} else str(output_norm)
        self.gamma_init = float(gamma_init)
        self.gamma_learnable = bool(gamma_learnable)
        self.merge_mode = str(merge_mode)
        self.merge_temperature = float(merge_temperature)
        self.merge_hidden_dim = int(merge_hidden_dim or self.d_model)
        self.merge_input_norm_type = None if merge_input_norm in {None, "none"} else str(merge_input_norm)
        self.merge_mean_bias_init = float(merge_mean_bias)
        self.eps = float(eps)

        self.ws1 = nn.Linear(self.d_model, self.attn_hidden_dim, bias=False)
        self.ws2 = nn.Linear(self.attn_hidden_dim, self.num_hops, bias=False)
        self.dropout = nn.Dropout(self.output_dropout)
        self.output_proj = nn.Linear(self.num_hops * self.d_model, self.d_model)
        if self.output_norm_type == "layernorm":
            self.output_norm = nn.LayerNorm(self.d_model)
        elif self.output_norm_type == "l2":
            self.output_norm = L2Norm(eps=self.eps)
        else:
            self.output_norm = None
        gamma_tensor = torch.tensor(self.gamma_init, dtype=torch.float32)
        if self.gamma_learnable:
            self.gamma = nn.Parameter(gamma_tensor)
        else:
            self.register_buffer("gamma", gamma_tensor)

        if self.merge_input_norm_type == "layernorm":
            self.merge_input_norm = nn.LayerNorm(self.d_model)
        elif self.merge_input_norm_type == "l2":
            self.merge_input_norm = L2Norm(eps=self.eps)
        else:
            self.merge_input_norm = None

        if self.merge_mode == "router":
            self.merge_router = nn.Sequential(
                nn.Linear(self.d_model, self.merge_hidden_dim),
                nn.GELU(),
                nn.Linear(self.merge_hidden_dim, 1),
            )
            self.merge_bias = nn.Parameter(
                torch.zeros(self.num_hops + 1, dtype=torch.float32)
            )
            if self.merge_mean_bias_init != 0.0:
                with torch.no_grad():
                    self.merge_bias[0] = self.merge_mean_bias_init
        else:
            self.merge_router = None
            self.merge_bias = None

        self.last_attention_weights = None
        self.last_structured_embedding = None

    def _compute_penalty(self, attn_weights: torch.Tensor) -> torch.Tensor:
        batch_size, num_hops, _ = attn_weights.shape
        identity = torch.eye(
            num_hops,
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        gram = torch.bmm(attn_weights, attn_weights.transpose(1, 2))
        penalty = (gram - identity).pow(2).sum(dim=(1, 2)).mean()
        return penalty

    def _masked_mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(-1)
        summed = torch.sum(hidden_states * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=self.eps)
        return summed / denom

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        batch_size, _, hidden_dim = hidden_states.shape
        if hidden_dim != self.d_model:
            raise ValueError(
                f"StructuredSelfAttentionPooling: expected hidden dim {self.d_model}, got {hidden_dim}"
            )

        mean_pooled = self._masked_mean_pool(hidden_states, attention_mask=attention_mask)
        scores = self.ws2(torch.tanh(self.ws1(hidden_states)))  # (B, L, r)
        scores = scores.transpose(1, 2)  # (B, r, L)

        if attention_mask is not None:
            mask = attention_mask.to(device=hidden_states.device)
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, r, L)
        structured_embedding = torch.bmm(attn_weights, hidden_states)  # (B, r, D)
        if self.merge_mode == "router":
            components = torch.cat(
                [mean_pooled.unsqueeze(1), structured_embedding],
                dim=1,
            )  # (B, r+1, D)
            routed_inputs = components
            if self.merge_input_norm is not None:
                routed_inputs = self.merge_input_norm(routed_inputs)
            router_scores = self.merge_router(routed_inputs).squeeze(-1)  # (B, r+1)
            router_scores = router_scores + self.merge_bias.to(
                device=router_scores.device, dtype=router_scores.dtype
            ).unsqueeze(0)
            router_weights = torch.softmax(router_scores / self.merge_temperature, dim=-1)
            pooled = torch.sum(components * router_weights.unsqueeze(-1), dim=1)
        else:
            residual = structured_embedding.reshape(batch_size, self.num_hops * self.d_model)
            residual = self.output_proj(self.dropout(residual))
            gamma = self.gamma.to(dtype=hidden_states.dtype, device=hidden_states.device)
            pooled = mean_pooled + gamma * residual
        if self.output_norm is not None:
            pooled = self.output_norm(pooled)

        self.last_attention_weights = attn_weights
        self.last_structured_embedding = structured_embedding
        penalty = self._compute_penalty(attn_weights)
        return pooled, penalty
