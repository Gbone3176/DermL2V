# bidirectional_qwen3.py

from typing import Optional, Tuple, Union, List

import torch
from torch import nn

from transformers import Qwen3Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging, add_start_docstrings_to_model_forward, replace_return_docstrings, can_return_tuple

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3PreTrainedModel,
    Qwen3Model,
    QWEN3_INPUTS_DOCSTRING,
)

logger = logging.get_logger(__name__)


class Qwen3BiModel(Qwen3Model):
    """
    双向版 Qwen3：去掉因果掩码，只保留 padding mask。
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)

        # 关键：关闭所有层的因果标志
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        output_attentions: bool = False,
    ):
        """
        覆盖原始的因果 mask 构造逻辑：
        - 如果没有 attention_mask：返回 None（完全全局注意力）
        - 如果是 2D mask：(B, L)，转换为 4D additive mask：(B, 1, L, L)
          只做 padding masking，不做时间方向的上三角遮挡
        """
        # 如果直接给了 4D mask，就原样返回（保持兼容）
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask

        if attention_mask is None:
            return None

        # attention_mask: (batch, seq_len)，值为 1/0
        batch_size, key_length = attention_mask.shape
        dtype = input_tensor.dtype
        device = input_tensor.device
        min_dtype = torch.finfo(dtype).min

        # 当前序列长度
        tgt_len = input_tensor.shape[1]

        # 只做 padding mask，不做上三角因果 mask：
        # 目标形状 (batch, 1, tgt_len, key_length)
        # 注意：这里不关心 cache_position / past_key_values，因为双向 MNTP 一般不用 KV cache
        mask_2d = attention_mask.to(device=device, dtype=input_tensor.dtype)  # 1 for keep, 0 for pad
        # 1 -> 0, 0 -> min_dtype
        additive_2d = (1.0 - mask_2d) * min_dtype
        # 扩展到 4D
        additive_4d = additive_2d[:, None, None, :].expand(batch_size, 1, tgt_len, key_length)

        return additive_4d


class Qwen3BiForMNTP(Qwen3PreTrainedModel):
    """
    Qwen3 双向 MNTP 模型：
    - backbone: Qwen3BiModel（全局双向注意力）
    - 头部: 线性 lm_head 做 token-level 预测
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    # ====== PEFT 兼容 ======
    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, peft_model):
        self.model = peft_model

    def save_peft_model(self, path: str):
        # 注意：这里假设 self.model 已经是 PeftModel
        self.model.save_pretrained(path)

    # ====== Embedding getter / setter，与 Qwen3ForCausalLM 保持一致 ======
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=Qwen3Config)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        MNTP 训练用 forward：
        - labels 形状 (batch, seq_len)，与 LM 一样，-100 的位置不计入 loss
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = False  # 双向 MNTP 一般不需要 KV cache，强制关掉更安全

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,          # 不使用 cache
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        logits = self.lm_head(hidden_states)      # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            # 和标准 LM 一样的 token-level CrossEntropy
            # labels: (batch, seq_len)，其中 -100 的位置会被忽略
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 展平成 (N, vocab) / (N,)
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
