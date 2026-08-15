import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=self.eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class StructuredSelfAttentionFusionPooling(nn.Module):
    """
    Structured self-attention pooling with a mean-pooling backbone and a routed
    multi-hop residual branch. The router only dispatches across the structured
    self-attention hop deltas relative to the mean-pooled view; the mean-pooled
    view is kept outside the routing softmax.
    """

    def __init__(
        self,
        d_model: int,
        attn_hidden_dim: int = 512,
        num_hops: int = 8,
        output_dropout: float = 0.0,
        output_norm: str | None = "layernorm",
        gamma_init: float = 1e-3,
        gamma_learnable: bool = True,
        merge_mode: str = "router",
        merge_temperature: float = 1.0,
        merge_hidden_dim: int | None = None,
        merge_input_norm: str | None = "layernorm",
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
        if merge_mode not in {"weighted_sum", "router"}:
            raise ValueError("merge_mode must be one of: 'weighted_sum', 'router'.")
        if output_norm not in {None, "none", "layernorm", "rmsnorm"}:
            raise ValueError(
                "output_norm must be one of: None, 'none', 'layernorm', 'rmsnorm'."
            )
        if merge_input_norm not in {None, "none", "layernorm", "l2"}:
            raise ValueError(
                "merge_input_norm must be one of: None, 'none', 'layernorm', 'l2'."
            )

        self.d_model = int(d_model)
        self.attn_hidden_dim = int(attn_hidden_dim)
        self.num_hops = int(num_hops)
        self.output_dropout = float(output_dropout)
        self.output_norm_type = None if output_norm in {None, "none"} else str(output_norm)
        self.gamma_init = float(gamma_init)
        self.gamma_learnable = bool(gamma_learnable)
        self.merge_mode = str(merge_mode)
        self.merge_temperature = float(merge_temperature)
        self.merge_hidden_dim = (
            int(merge_hidden_dim)
            if merge_hidden_dim is not None
            else max(64, self.d_model // 8)
        )
        self.merge_input_norm_type = (
            None if merge_input_norm in {None, "none"} else str(merge_input_norm)
        )
        self.eps = float(eps)

        self.ws1 = nn.Linear(self.d_model, self.attn_hidden_dim, bias=False)
        self.ws2 = nn.Linear(self.attn_hidden_dim, self.num_hops, bias=False)
        self.dropout = nn.Dropout(self.output_dropout)
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
            nn.init.zeros_(self.merge_router[-1].weight)
            nn.init.zeros_(self.merge_router[-1].bias)
        else:
            self.merge_router = None

        # Kept for config backward compatibility. Routing now only happens across
        # the structured SA hops, so there is no separate mean-view bias term.
        self.merge_bias = nn.Parameter(torch.zeros(self.num_hops))

        if self.output_norm_type == "layernorm":
            self.output_norm = nn.LayerNorm(self.d_model)
        elif self.output_norm_type == "rmsnorm":
            self.output_norm = RMSNorm(self.d_model, eps=self.eps)
        else:
            self.output_norm = None

        self.last_attention_weights = None
        self.last_structured_embedding = None
        self.last_delta_embedding = None
        self.last_merge_weights = None

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

        mask = attention_mask.to(
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).unsqueeze(-1)
        summed = torch.sum(hidden_states * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=self.eps)
        return summed / denom

    def _apply_input_norm(self, views: torch.Tensor) -> torch.Tensor:
        if self.merge_input_norm is None:
            return views
        return self.merge_input_norm(views)

    def _compute_merge_scores(self, views: torch.Tensor) -> torch.Tensor:
        temperature = max(self.merge_temperature, self.eps)
        if self.merge_mode == "weighted_sum":
            scores = views.pow(2).sum(dim=-1) / temperature
        else:
            flat_views = views.reshape(-1, self.d_model)
            scores = self.merge_router(flat_views).reshape(views.size(0), views.size(1))
            scores = scores / temperature
        return scores + self.merge_bias.unsqueeze(0)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None):
        batch_size, _, hidden_dim = hidden_states.shape
        if hidden_dim != self.d_model:
            raise ValueError(
                f"StructuredSelfAttentionFusionPooling: expected hidden dim {self.d_model}, got {hidden_dim}"
            )

        mean_pooled = self._masked_mean_pool(hidden_states, attention_mask=attention_mask)
        scores = self.ws2(torch.tanh(self.ws1(hidden_states)))  # (B, L, r)
        scores = scores.transpose(1, 2)  # (B, r, L)

        if attention_mask is not None:
            mask = attention_mask.to(device=hidden_states.device)
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min)

        attn_weights = torch.softmax(scores, dim=-1)  # (B, r, L)
        structured_embedding = torch.bmm(attn_weights, hidden_states)  # (B, r, D)
        structured_embedding = self.dropout(structured_embedding)

        delta_embedding = structured_embedding - mean_pooled.unsqueeze(1)
        routed_views = self._apply_input_norm(delta_embedding)
        merge_scores = self._compute_merge_scores(routed_views)
        merge_weights = torch.softmax(merge_scores, dim=-1)
        fused_delta = torch.sum(
            delta_embedding * merge_weights.unsqueeze(-1), dim=1
        )
        gamma = self.gamma.to(dtype=hidden_states.dtype, device=hidden_states.device)
        pooled = mean_pooled + gamma * fused_delta

        if self.output_norm is not None:
            pooled = self.output_norm(pooled)

        self.last_attention_weights = attn_weights
        self.last_structured_embedding = structured_embedding
        self.last_delta_embedding = delta_embedding
        self.last_merge_weights = merge_weights
        penalty = self._compute_penalty(attn_weights)
        return pooled, penalty
