"""
NV-Embed style Latent Attention Pooling for LLM2Vec (interface compatible).

V3 improvements over V2:
- [Fix #2] Use F.scaled_dot_product_attention (SDPA) for FlashAttention / MemoryEfficient backend
- [Fix #3] Add dropout to attention, FFN, and residual connections
- [Fix #4] Default head_dim = d_model // num_heads so inner dim matches d_model
- [Fix #5] Use trunc_normal_ for latent initialization instead of randn * 0.02

Key properties:
- Input: hidden_states (B, L, D), attention_mask (B, L) with 1=keep, 0=mask.
- Output: pooled (B, D)
- Implements: PreNorm + Cross-Attn (Q=tokens, K=V=latents) + Residual
             + PreNorm + FFN + Residual
             + masked mean pooling over tokens
- Does NOT require changing LLM2Vec.get_pooling() call signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class _FeedForward(nn.Module):
    """
    NV-like FFN with GEGLU:
    Linear(D -> 2*mult*D) -> GEGLU -> Dropout -> Linear(mult*D -> D) -> Dropout
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner * 2, bias=True),
            _GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PreNorm(nn.Module):
    """
    PreNorm wrapper:
    y = fn(LN(x), LN(context))
    """
    def __init__(self, dim: int, fn: nn.Module, context_dim: int | None = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x: torch.Tensor, *, context: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        if self.norm_context is not None and context is not None:
            context = self.norm_context(context)
            return self.fn(x, context=context, **kwargs)
        # If no context path, call fn without unexpected kwarg
        return self.fn(x, **kwargs)


class _CrossAttention(nn.Module):
    """
    Cross attention where:
    - Q from x (tokens): (B, L, D)
    - K,V from context (latents): (B, R, D)

    Uses F.scaled_dot_product_attention for automatic FlashAttention / MemoryEfficient dispatch.
    """
    def __init__(self, dim: int, num_heads: int = 8, head_dim: int | None = None, attn_dropout: float = 0.0):
        super().__init__()
        # Fix #4: default head_dim = dim // num_heads so that inner == dim
        if head_dim is None:
            head_dim = dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor, *, context: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, L, D)         -> queries
        context: (B, R, D)   -> keys/values (latents)
        attn_bias: optional additive bias broadcastable to (B, H, L, R)
        """
        B, L, _ = x.shape
        _, R, _ = context.shape
        H = self.num_heads
        Dh = self.head_dim

        q = self.to_q(x).view(B, L, H, Dh).transpose(1, 2)         # (B, H, L, Dh)
        k = self.to_k(context).view(B, R, H, Dh).transpose(1, 2)   # (B, H, R, Dh)
        v = self.to_v(context).view(B, R, H, Dh).transpose(1, 2)   # (B, H, R, Dh)

        # Fix #2: Use SDPA for FlashAttention / MemoryEfficient backend
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,       # additive bias (broadcastable to B, H, L, R)
            dropout_p=dropout_p,
            is_causal=False,
        )  # (B, H, L, Dh)

        out = out.transpose(1, 2).contiguous().view(B, L, H * Dh)  # (B, L, inner)
        return self.to_out(out)  # (B, L, D)


class LatentAttentionPooling(nn.Module):
    """
    NV-Embed style latent attention pooling (token->latent cross attention + residual + FFN + residual + mean pool).

    Interface-compatible with your current usage:
        pooled = latent_attn(hidden_states, attention_mask=mask)
    """
    def __init__(
        self,
        d_model: int,
        num_latents: int = 512,
        num_heads: int = 8,
        head_dim: int | None = None,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        output_normalize: bool = False,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        self.output_normalize = output_normalize
        self.eps = eps

        # Fix #5: Use truncated normal for better latent initialization
        self.latents = nn.Parameter(torch.empty(num_latents, d_model))
        nn.init.trunc_normal_(self.latents, std=0.02)

        # Fix #4: head_dim defaults to d_model // num_heads (passed through to _CrossAttention)
        # Fix #3: dropout params forwarded to attention and FFN
        self.cross_attn = _PreNorm(
            d_model,
            _CrossAttention(d_model, num_heads=num_heads, head_dim=head_dim, attn_dropout=attn_dropout),
            context_dim=d_model,
        )
        self.cross_ff = _PreNorm(
            d_model,
            _FeedForward(d_model, mult=ff_mult, dropout=ff_dropout),
            context_dim=None,
        )

        # Fix #3: Residual dropout (applied after each sub-layer before adding residual)
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, D)
            attention_mask: (B, L) with 1=keep, 0=mask (pooling mask)
        Returns:
            pooled: (B, D)
        """
        B, L, D = hidden_states.shape
        if D != self.d_model:
            raise ValueError(f"LatentAttentionPooling: expected hidden dim {self.d_model}, got {D}")

        device = hidden_states.device
        latents = self.latents.unsqueeze(0).expand(B, -1, -1).to(device)  # (B, R, D)

        # --- Cross-attention block with residual ---
        # Tokens(Q) attend Latents(K=V)
        h = self.cross_attn(hidden_states, context=latents)
        hidden_states = hidden_states + self.residual_dropout(h)

        # --- FeedForward block with residual ---
        h2 = self.cross_ff(hidden_states)
        hidden_states = hidden_states + self.residual_dropout(h2)

        # --- Masked mean pooling over tokens ---
        if attention_mask is not None:
            # attention_mask: (B, L), 1 keep, 0 mask
            mask = attention_mask.to(dtype=hidden_states.dtype, device=device).unsqueeze(-1)  # (B, L, 1)
            summed = torch.sum(hidden_states * mask, dim=1)  # (B, D)
            denom = torch.clamp(mask.sum(dim=1), min=self.eps)  # (B, 1)
            pooled = summed / denom
        else:
            pooled = hidden_states.mean(dim=1)

        if self.output_normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled
