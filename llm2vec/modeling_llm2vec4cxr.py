"""
Custom model class for LLM2Vec4CXR that properly handles latent attention pooling.
"""

from llm2vec.models.bidirectional_llama import LlamaBiModel
from transformers import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
# from llm2vec.pooling import LatentAttentionPooling
from .pooling_latent import LatentAttentionPooling
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class LLM2Vec4CXRModel(PreTrainedModel):
    """
    Wrapper model that includes LlamaBiModel and latent attention pooling.
    Structure matches the saved checkpoint: self.model + self.latent_attn
    """
    config_class = LlamaConfig
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        # Wrap the LlamaBiModel
        self.model = LlamaBiModel(config)
        
        # Initialize latent attention pooling
        self.latent_attn = LatentAttentionPooling(
            d_model=config.hidden_size,
            num_heads=8,  # Standard for this model size
            num_latents=512  # Standard for LLM2Vec
        )
    
    def forward(self, input_ids, attention_mask=None, embed_mask=None, **kwargs):
        """
        Forward pass that properly handles latent attention pooling.
        """
        # Get base model output
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        
        # Apply latent attention pooling
        if embed_mask is not None:
            # Use embed_mask for instruction-following tasks
            pooled_output = self.latent_attn(outputs.last_hidden_state, embed_mask)
        else:
            # Use attention_mask for simple encoding
            pooled_output = self.latent_attn(outputs.last_hidden_state, attention_mask)
        
        return pooled_output

    # --- Convenience tokenizer (lazy) -------------------------------------
    def _get_tokenizer(self):
        if not hasattr(self, "_hf_tokenizer"):
            tok = AutoTokenizer.from_pretrained(getattr(self.config, "_name_or_path", "lukeingawesome/llm2vec4cxr"))
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            self._hf_tokenizer = tok
        return self._hf_tokenizer

    # --- Ensure latent_attn follows .to(device/dtype) ----------------------
    def to(self, *args, **kwargs):
        m = super().to(*args, **kwargs)
        if hasattr(self, "latent_attn") and self.latent_attn is not None:
            # Align latent_attn with the base weights' device & dtype
            try:
                device = next(p.device for p in self.parameters() if p is not None)
                dtype  = next((p.dtype for p in self.parameters() if p.is_floating_point()), None)
                self.latent_attn = self.latent_attn.to(device=device, dtype=dtype)
            except StopIteration:
                pass
        return m

    # --- Simple text encoding (no instruction) ----------------------------
    @torch.no_grad()
    def encode_text(self, texts, max_length: int = 512):
        tok = self._get_tokenizer()
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        # For simple encoding we embed over all non‑pad tokens
        enc["embed_mask"] = enc["attention_mask"].clone()
        dev = next(self.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        return self(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], embed_mask=enc["embed_mask"])

    # --- Instruction/text encoding with separator -------------------------
    def _build_separator_inputs(self, texts, max_length: int, separator: str):
        tok = self._get_tokenizer()
        # Split into [instruction | text]; we embed only the trailing "text" part.
        # If no separator, embed the entire text.
        parts_after_sep = []
        original = []
        for t in texts:
            parts = t.split(separator)
            # If no separator found, use the entire text (not empty string)
            parts_after_sep.append(parts[1] if len(parts) > 1 else parts[0])
            original.append("".join(parts))

        tokenized = tok(original, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        # Build an embed_mask that lights up only the trailing "text" span
        embed_mask = None
        for i, t in enumerate(parts_after_sep):
            sub = tok([t], return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)
            m = torch.zeros_like(tokenized["attention_mask"][i])
            if len(sub["input_ids"][0]) > 0:
                m[-len(sub["input_ids"][0]):] = 1
            else:
                # If tokenization resulted in 0 tokens, use attention_mask (embed everything)
                m = tokenized["attention_mask"][i].clone()
            embed_mask = m.unsqueeze(0) if embed_mask is None else torch.cat([embed_mask, m.unsqueeze(0)], dim=0)

        tokenized["embed_mask"] = embed_mask
        return tokenized

    @torch.no_grad()
    def encode_with_separator(self, texts, separator: str = "!@#$%^&*()", max_length: int = 512):
        enc = self._build_separator_inputs(texts, max_length=max_length, separator=separator)
        dev = next(self.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        return self(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], embed_mask=enc["embed_mask"])

    # --- One‑liner cosine similarity over instruction+text ----------------
    @torch.no_grad()
    def compute_similarities(self, query_text: str, candidate_texts, separator: str = "!@#$%^&*()", max_length: int = 512):
        all_texts = [query_text] + list(candidate_texts)
        embs = self.encode_with_separator(all_texts, separator=separator, max_length=max_length)
        # embs: [N, 2048]; compare query vs candidates
        return F.cosine_similarity(embs[0], embs[1:], dim=1)
