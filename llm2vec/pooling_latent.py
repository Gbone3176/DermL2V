"""
Latent Attention Pooling implementation for LLM2Vec4CXR.
Vendored to make the model self-contained (no external llm2vec dependency required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentAttentionPooling(nn.Module):
    """
    Latent attention pooling layer that uses a trainable latent dictionary
    to aggregate token embeddings into a fixed-size representation.
    """
    
    def __init__(self, d_model, num_latents=512, num_heads=8):
        """
        Args:
            d_model: Hidden size of the model (e.g., 2048 for Llama-7B)
            num_latents: Number of learnable latent vectors (default: 512)
            num_heads: Number of attention heads (default: 8)
        """
        super().__init__()
        self.num_latents = num_latents
        self.d_model = d_model
        
        # Trainable latent dictionary (used as both keys and values)
        self.latents = nn.Parameter(torch.empty(num_latents, d_model))
        nn.init.trunc_normal_(self.latents, std=0.02)
        
        # Multihead attention layer
        # batch_first=True means input shape is (batch, seq_length, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Simple MLP: Linear -> GELU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Apply latent attention pooling to hidden states.
        
        Args:
            hidden_states: Token embeddings of shape (batch_size, seq_len, d_model)
            attention_mask: Optional mask of shape (batch_size, seq_len)
        
        Returns:
            Pooled embeddings of shape (batch_size, d_model)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        if d_model != self.d_model:
            raise ValueError(
                f"LatentAttentionPooling: expected hidden dim {self.d_model}, got {d_model}"
            )
        
        # Expand latents to match batch size: (batch_size, num_latents, d_model)
        # Never call `self.to(...)` inside forward in DDP/FSDP training.
        # Cast a view of latents to current hidden_states device/dtype instead.
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        # Apply multihead attention
        # Use hidden_states as queries and latent dictionary as keys/values
        # This computes: O = softmax((QK^T)/√d)V
        attn_output, _ = self.multihead_attn(
            query=hidden_states, 
            key=latents, 
            value=latents,
            need_weights=False,
        )
        attn_output = self.attn_norm(attn_output)
        residual_output = hidden_states + attn_output
        mlp_output = self.mlp(residual_output)
        
        # Mean pool over sequence dimension
        if attention_mask is not None:
            # Mask out padding tokens before pooling
            mask_expanded = (
                attention_mask.to(dtype=mlp_output.dtype, device=mlp_output.device)
                .unsqueeze(-1)
                .expand(mlp_output.size())
            )
            sum_embeddings = torch.sum(mlp_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # Simple mean pooling if no mask provided
            pooled = mlp_output.mean(dim=1)
        
        return pooled
