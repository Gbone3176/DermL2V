"""
LLM2Vec V4 memory token prepend variant.

This variant is built from the original V1 code path and adds a training-free
cross-layer sentence-memory mechanism for bidirectional embedding models.

Core idea:
- compute a layer feature (LF) after each transformer layer via masked mean pooling
- prepend the previous layer's LF to the next layer as a memory token
- let later layers directly attend to an explicit sentence-level summary
- exclude prepended memory tokens from the final pooling by default

Current scope:
- keeps the original V1 loading/tokenization/encoding interface as much as possible
- implements the cross-layer LF prepend path only for `LlamaBiModel`
- does not modify any existing V1/V3 files or require additional training
"""

import json
import logging
import os
from typing import Dict, Optional, Union

import torch
from peft import PeftModel
from torch import Tensor
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

from .llm2vecV1 import LLM2Vec as LLM2VecV1
from .models import LlamaBiModel

logger = logging.getLogger(__name__)


class LLM2Vec(LLM2VecV1):
    def __init__(
        self,
        *args,
        cross_layer_lf_prepend: bool = False,
        cross_layer_lf_start_layer: int = 0,
        cross_layer_lf_end_layer: Optional[int] = None,
        cross_layer_lf_exclude_from_pooling: bool = True,
        cross_layer_lf_use_embed_mask: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cross_layer_lf_prepend = cross_layer_lf_prepend
        self.cross_layer_lf_start_layer = max(0, int(cross_layer_lf_start_layer))
        self.cross_layer_lf_end_layer = (
            None
            if cross_layer_lf_end_layer is None
            else max(self.cross_layer_lf_start_layer, int(cross_layer_lf_end_layer))
        )
        self.cross_layer_lf_exclude_from_pooling = cross_layer_lf_exclude_from_pooling
        self.cross_layer_lf_use_embed_mask = cross_layer_lf_use_embed_mask

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path,
        peft_model_name_or_path=None,
        merge_peft=True,
        enable_bidirectional=True,
        extra_model_name_or_path=None,
        use_safetensors: bool = True,
        **kwargs,
    ):
        keys = [
            "pooling_mode",
            "max_length",
            "doc_max_length",
            "skip_instruction",
            "cross_layer_lf_prepend",
            "cross_layer_lf_start_layer",
            "cross_layer_lf_end_layer",
            "cross_layer_lf_exclude_from_pooling",
            "cross_layer_lf_use_embed_mask",
        ]
        # Remove V4-specific encoder kwargs before forwarding to the HF backbone.
        # Some of them are intentionally allowed to be None, so filtering on
        # value would leak unknown kwargs such as cross_layer_lf_end_layer=None.
        encoder_args = {}
        for key in keys:
            if key in kwargs:
                value = kwargs.pop(key)
                if value is not None:
                    encoder_args[key] = value

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_name_or_path)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )
        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)
        logger.info("Loaded base model from %s", base_model_name_or_path)

        if os.path.isdir(base_model_name_or_path) and os.path.exists(
            f"{base_model_name_or_path}/config.json"
        ):
            with open(f"{base_model_name_or_path}/config.json", "r") as f_in:
                config_dict = json.load(f_in)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(model, base_model_name_or_path)
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(model, peft_model_name_or_path)
            if merge_peft:
                model = model.merge_and_unload()
                logger.info(
                    "have merged PEFT adapter weights from %s",
                    peft_model_name_or_path,
                )
        else:
            logger.info("PEFT model is not provided.")

        if extra_model_name_or_path is not None and len(extra_model_name_or_path) > 0:
            if not merge_peft and isinstance(model, PeftModel):
                model = model.merge_and_unload()
                logger.info(
                    "have merged base model weights from %s", base_model_name_or_path
                )
            if isinstance(extra_model_name_or_path, str):
                model = PeftModel.from_pretrained(model, extra_model_name_or_path)
                model = model.merge_and_unload()
                logger.info(
                    "have merged extra PEFT adapter weights from %s",
                    extra_model_name_or_path,
                )
                peft_model_name_or_path = extra_model_name_or_path
            elif isinstance(extra_model_name_or_path, list):
                for extra_model in extra_model_name_or_path:
                    model = PeftModel.from_pretrained(model, extra_model)
                    model = model.merge_and_unload()
                    peft_model_name_or_path = extra_model
                    logger.info(
                        "have merged extra PEFT adapter weights from %s", extra_model
                    )
            else:
                raise ValueError(
                    "extra_model_name_or_path should be a string or a list of strings."
                )
        else:
            logger.info("extra model is not provided.")

        resolved_config = {}
        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as f_in:
                llm2vec_config = json.load(f_in)
            resolved_config.update(llm2vec_config)

        for key, value in encoder_args.items():
            resolved_config[key] = value

        llm2vec_model = cls(model=model, tokenizer=tokenizer, **resolved_config)

        if (
            getattr(llm2vec_model, "latent_attn", None) is not None
            and llm2vec_model.pooling_mode == "latent_pooling"
        ):
            llm2vec_model._load_latent_attention_weights(
                peft_model_name_or_path, use_safetensors=use_safetensors
            )

        if "torch_dtype" in kwargs and kwargs["torch_dtype"] is not None:
            llm2vec_model = llm2vec_model.to(kwargs["torch_dtype"])  # type: ignore

        return llm2vec_model

    def _masked_mean_pooling(
        self, hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden = hidden_states * mask
        denom = torch.clamp(mask.sum(dim=1), min=1e-9)
        return masked_hidden.sum(dim=1) / denom

    def _validate_cross_layer_lf_setup(self):
        if not self.cross_layer_lf_prepend:
            return
        if self.pooling_mode == "bos_token":
            raise ValueError(
                "cross_layer_lf_prepend does not support pooling_mode='bos_token'."
            )
        if isinstance(self.model, PeftModel):
            raise ValueError(
                "cross_layer_lf_prepend expects a merged backbone, but self.model is still a PeftModel."
            )
        if not isinstance(self.model, LlamaBiModel):
            raise ValueError(
                "cross_layer_lf_prepend is currently implemented only for LlamaBiModel."
            )

    def _prepend_bool_mask(
        self,
        mask: Optional[Tensor],
        value: int,
    ) -> Optional[Tensor]:
        if mask is None:
            return None
        prefix = mask.new_full((mask.shape[0], 1), value)
        return torch.cat([prefix, mask], dim=1)

    def _prepend_pad_token_id(self, input_ids: Optional[Tensor]) -> Optional[Tensor]:
        if input_ids is None:
            return None
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        prefix = input_ids.new_full((input_ids.shape[0], 1), pad_token_id)
        return torch.cat([prefix, input_ids], dim=1)

    def _get_lf_mask(
        self,
        attention_mask: Tensor,
        embed_mask: Optional[Tensor],
    ) -> Tensor:
        if self.cross_layer_lf_use_embed_mask and embed_mask is not None:
            return embed_mask
        return attention_mask

    def _run_backbone(
        self,
        sentence_feature: Dict[str, Tensor],
        output_hidden_states: bool = False,
    ):
        embed_mask = sentence_feature.get("embed_mask")
        model_inputs = {
            k: v for k, v in sentence_feature.items() if k not in {"embed_mask"}
        }

        if not self.cross_layer_lf_prepend:
            if output_hidden_states:
                model_inputs["output_hidden_states"] = True
            reps = self.model(**model_inputs)
            return {
                "last_hidden_state": reps.last_hidden_state,
                "hidden_states": getattr(reps, "hidden_states", None),
                "pool_attention_mask": sentence_feature["attention_mask"],
                "pool_input_ids": sentence_feature.get("input_ids"),
                "pool_embed_mask": embed_mask,
            }

        self._validate_cross_layer_lf_setup()
        return self._run_llama_cross_layer_lf(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
            embed_mask=embed_mask,
            inputs_embeds=model_inputs.get("inputs_embeds"),
            output_hidden_states=output_hidden_states,
        )

    def _run_llama_cross_layer_lf(
        self,
        input_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        embed_mask: Optional[Tensor],
        inputs_embeds: Optional[Tensor],
        output_hidden_states: bool,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Exactly one of input_ids or inputs_embeds must be provided."
            )

        if inputs_embeds is None:
            hidden_states = self.model.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.long,
                device=hidden_states.device,
            )

        lf_mask = self._get_lf_mask(attention_mask, embed_mask)
        final_pool_mask = (
            lf_mask.clone()
            if self.cross_layer_lf_exclude_from_pooling
            else attention_mask.clone()
        )
        tracked_input_ids = input_ids

        all_hidden_states = () if output_hidden_states else None
        num_layers = len(self.model.layers)
        inject_end = (
            num_layers - 1
            if self.cross_layer_lf_end_layer is None
            else min(self.cross_layer_lf_end_layer, num_layers - 1)
        )

        for layer_idx, decoder_layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            cache_position = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            )
            position_ids = cache_position.unsqueeze(0)
            causal_mask = self.model._update_causal_mask(
                attention_mask=attention_mask,
                input_tensor=hidden_states,
                cache_position=cache_position,
                past_key_values=None,
                output_attentions=False,
            )
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            should_inject = (
                layer_idx >= self.cross_layer_lf_start_layer
                and layer_idx < inject_end
            )
            if should_inject:
                layer_feature = self._masked_mean_pooling(hidden_states, lf_mask)
                hidden_states = torch.cat(
                    [layer_feature.unsqueeze(1), hidden_states], dim=1
                )
                attention_mask = self._prepend_bool_mask(attention_mask, value=1)
                lf_mask = self._prepend_bool_mask(lf_mask, value=0)
                final_value = 0 if self.cross_layer_lf_exclude_from_pooling else 1
                final_pool_mask = self._prepend_bool_mask(
                    final_pool_mask, value=final_value
                )
                tracked_input_ids = self._prepend_pad_token_id(tracked_input_ids)

        hidden_states = self.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        pool_embed_mask = (
            final_pool_mask
            if self.skip_instruction or embed_mask is not None
            else None
        )
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "pool_attention_mask": final_pool_mask,
            "pool_input_ids": tracked_input_ids,
            "pool_embed_mask": pool_embed_mask,
        }

    def forward(self, sentence_feature: Dict[str, Tensor]):
        backbone_state = self._run_backbone(sentence_feature, output_hidden_states=False)
        pool_features = {
            "attention_mask": backbone_state["pool_attention_mask"],
        }
        if backbone_state["pool_input_ids"] is not None:
            pool_features["input_ids"] = backbone_state["pool_input_ids"]
        if backbone_state["pool_embed_mask"] is not None:
            pool_features["embed_mask"] = backbone_state["pool_embed_mask"]
        return self.get_pooling(pool_features, backbone_state["last_hidden_state"])

    def convert_to_bert_format(
        self,
        sentence_feature: Dict[str, Tensor],
        pooling: Optional[str] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, Tensor]:
        if return_tensors != "pt":
            raise ValueError(
                "Only return_tensors='pt' is supported for convert_to_bert_format()."
            )

        selected_pooling = pooling if pooling is not None else self.pooling_mode
        backbone_state = self._run_backbone(
            sentence_feature,
            output_hidden_states=False,
        )
        last_hidden = backbone_state["last_hidden_state"].to(torch.float32)

        tmp_features = {"attention_mask": backbone_state["pool_attention_mask"]}
        if backbone_state["pool_input_ids"] is not None:
            tmp_features["input_ids"] = backbone_state["pool_input_ids"]
        if backbone_state["pool_embed_mask"] is not None:
            tmp_features["embed_mask"] = backbone_state["pool_embed_mask"]

        prev_mode = self.pooling_mode
        try:
            self.pooling_mode = selected_pooling
            pooled = self.get_pooling(tmp_features, last_hidden).to(torch.float32)
        finally:
            self.pooling_mode = prev_mode

        return {
            "last_hidden_state": last_hidden,
            "pooler_output": pooled,
        }

    def save(self, output_path, merge_before_save=False, save_config=True):
        if merge_before_save and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            if hasattr(self.model, "_hf_peft_config_loaded"):
                self.model._hf_peft_config_loaded = False

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        llm2vec_config = {
            "pooling_mode": self.pooling_mode,
            "max_length": self.max_length,
            "doc_max_length": self.doc_max_length,
            "skip_instruction": self.skip_instruction,
            "cross_layer_lf_prepend": self.cross_layer_lf_prepend,
            "cross_layer_lf_start_layer": self.cross_layer_lf_start_layer,
            "cross_layer_lf_end_layer": self.cross_layer_lf_end_layer,
            "cross_layer_lf_exclude_from_pooling": self.cross_layer_lf_exclude_from_pooling,
            "cross_layer_lf_use_embed_mask": self.cross_layer_lf_use_embed_mask,
        }

        if save_config:
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/llm2vec_config.json", "w") as f_out:
                json.dump(llm2vec_config, f_out, indent=4)

        if getattr(self, "latent_attn", None) is not None:
            try:
                torch.save(
                    self.latent_attn.state_dict(),
                    os.path.join(output_path, "latent_attn.pt"),
                )
            except Exception as e:
                logger.warning("Failed to save latent_attn weights: %s", e)
