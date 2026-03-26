import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPPooling(nn.Module):
    """
    A lightweight pooling head that applies a small residual MLP on top of token
    embeddings before masked mean pooling.

    Design goal:
    - preserve the original embedding space at initialization
    - allow gradual domain correction during fine-tuning
    - avoid injecting randomly initialized latent prototypes
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        num_layers: int = 4,
        dropout: float = 0.0,
        gamma_init: float = 1e-3,
        gamma_learnable: bool = True,
        output_normalize: bool = False,
        output_layernorm: bool = False,
        eps: float = 1e-9,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("ResidualMLPPooling expects num_layers >= 2.")

        self.d_model = d_model
        self.hidden_dim = hidden_dim if hidden_dim is not None else d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_normalize = output_normalize
        self.output_layernorm = output_layernorm
        self.eps = eps

        self.input_norm = nn.LayerNorm(d_model)
        self.pool_norm = nn.LayerNorm(d_model) if output_layernorm else None

        layers = []
        in_dim = d_model
        for layer_idx in range(num_layers):
            out_dim = d_model if layer_idx == num_layers - 1 else self.hidden_dim
            linear = nn.Linear(in_dim, out_dim)
            if layer_idx == num_layers - 1:
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            if layer_idx != num_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = self.hidden_dim
        self.mlp = nn.Sequential(*layers)

        gamma_tensor = torch.tensor(float(gamma_init), dtype=torch.float32)
        if gamma_learnable:
            self.gamma = nn.Parameter(gamma_tensor)
        else:
            self.register_buffer("gamma", gamma_tensor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.size(-1) != self.d_model:
            raise ValueError(
                f"ResidualMLPPooling expected hidden dim {self.d_model}, got {hidden_states.size(-1)}"
            )

        normalized = self.input_norm(hidden_states)
        delta = self.mlp(normalized)
        gamma = self.gamma.to(dtype=hidden_states.dtype, device=hidden_states.device)
        refined = hidden_states + gamma * delta

        if attention_mask is not None:
            mask = attention_mask.to(dtype=refined.dtype, device=refined.device).unsqueeze(-1)
            summed = torch.sum(refined * mask, dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=self.eps)
            pooled = summed / denom
        else:
            pooled = refined.mean(dim=1)

        if self.pool_norm is not None:
            pooled = self.pool_norm(pooled)
        if self.output_normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled
