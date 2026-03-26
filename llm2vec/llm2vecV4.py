"""
LLM2Vec V5 ResMLP Pooling with less parameters to refine the embedding space.
"""

import json
import os
import logging
from typing import Optional

import torch

from .llm2vecV1 import LLM2Vec as LLM2VecV1
from .pooling_residual_mlp import ResidualMLPPooling


logger = logging.getLogger(__name__)


class LLM2Vec(LLM2VecV1):
    """
    V4 is V1 plus an optional residual MLP pooling head.
    """

    def __init__(
        self,
        model,
        tokenizer,
        pooling_mode: str = "mean",
        max_length: int = 512,
        doc_max_length: int = 400,
        skip_instruction: bool = True,
        res_mlp_hidden_dim: Optional[int] = None,
        res_mlp_num_layers: int = 4,
        res_mlp_dropout: float = 0.0,
        res_mlp_gamma_init: float = 1e-3,
        res_mlp_gamma_learnable: bool = True,
        res_mlp_output_normalize: bool = False,
        res_mlp_output_layernorm: bool = False,
    ):
        init_pooling_mode = "mean" if pooling_mode == "res_mlp_pooling" else pooling_mode
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            pooling_mode=init_pooling_mode,
            max_length=max_length,
            doc_max_length=doc_max_length,
            skip_instruction=skip_instruction,
        )

        self.res_mlp_hidden_dim = res_mlp_hidden_dim
        self.res_mlp_num_layers = res_mlp_num_layers
        self.res_mlp_dropout = res_mlp_dropout
        self.res_mlp_gamma_init = res_mlp_gamma_init
        self.res_mlp_gamma_learnable = res_mlp_gamma_learnable
        self.res_mlp_output_normalize = res_mlp_output_normalize
        self.res_mlp_output_layernorm = res_mlp_output_layernorm

        self.res_mlp_pooler: Optional[ResidualMLPPooling] = None
        if pooling_mode == "res_mlp_pooling":
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(
                    "Model config must define hidden_size to use res_mlp_pooling."
                )
            self.res_mlp_pooler = ResidualMLPPooling(
                d_model=hidden_size,
                hidden_dim=res_mlp_hidden_dim,
                num_layers=res_mlp_num_layers,
                dropout=res_mlp_dropout,
                gamma_init=res_mlp_gamma_init,
                gamma_learnable=res_mlp_gamma_learnable,
                output_normalize=res_mlp_output_normalize,
                output_layernorm=res_mlp_output_layernorm,
            )
            self.pooling_mode = pooling_mode

    def to(self, device_or_dtype):
        result = super().to(device_or_dtype)
        if getattr(result, "res_mlp_pooler", None) is not None:
            result.res_mlp_pooler = result.res_mlp_pooler.to(device_or_dtype)
        return result

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
        extra_keys = [
            "res_mlp_hidden_dim",
            "res_mlp_num_layers",
            "res_mlp_dropout",
            "res_mlp_gamma_init",
            "res_mlp_gamma_learnable",
            "res_mlp_output_normalize",
            "res_mlp_output_layernorm",
        ]
        extra_encoder_args = {
            key: kwargs.pop(key, None) for key in extra_keys if kwargs.get(key) is not None
        }

        model = super().from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            merge_peft=merge_peft,
            enable_bidirectional=enable_bidirectional,
            extra_model_name_or_path=extra_model_name_or_path,
            use_safetensors=use_safetensors,
            **kwargs,
        )

        for key, value in extra_encoder_args.items():
            setattr(model, key, value)

        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if (
            getattr(model, "res_mlp_pooler", None) is not None
            and getattr(model, "pooling_mode", None) == "res_mlp_pooling"
        ):
            model._load_res_mlp_weights(config_addr)

        return model

    def get_pooling(self, features, last_hidden_states, hidden_states=None):
        if self.pooling_mode == "res_mlp_pooling":
            if self.res_mlp_pooler is None:
                raise RuntimeError(
                    "res_mlp_pooler is not initialized but res_mlp_pooling was selected."
                )
            attn_mask = None
            if "embed_mask" in features and features["embed_mask"] is not None:
                attn_mask = features["embed_mask"].to(last_hidden_states.device)
            else:
                attn_mask = features.get("attention_mask", None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(last_hidden_states.device)
            return self.res_mlp_pooler(last_hidden_states, attention_mask=attn_mask)
        return super().get_pooling(features, last_hidden_states)

    def save(self, output_path, merge_before_save=False, save_config=True):
        super().save(
            output_path=output_path,
            merge_before_save=merge_before_save,
            save_config=save_config,
        )

        if save_config:
            config_path = os.path.join(output_path, "llm2vec_config.json")
            config = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f_in:
                    config = json.load(f_in)
            config.update(
                {
                    "res_mlp_hidden_dim": self.res_mlp_hidden_dim,
                    "res_mlp_num_layers": self.res_mlp_num_layers,
                    "res_mlp_dropout": self.res_mlp_dropout,
                    "res_mlp_gamma_init": self.res_mlp_gamma_init,
                    "res_mlp_gamma_learnable": self.res_mlp_gamma_learnable,
                    "res_mlp_output_normalize": self.res_mlp_output_normalize,
                    "res_mlp_output_layernorm": self.res_mlp_output_layernorm,
                }
            )
            with open(config_path, "w") as f_out:
                json.dump(config, f_out, indent=4)

        if self.res_mlp_pooler is not None:
            try:
                torch.save(
                    self.res_mlp_pooler.state_dict(),
                    os.path.join(output_path, "res_mlp_pooler.pt"),
                )
            except Exception as e:
                logger.warning(f"Failed to save res_mlp_pooler weights: {e}")

    def _load_res_mlp_weights(self, model_path: Optional[str]):
        if self.res_mlp_pooler is None or model_path is None:
            return
        try:
            state_path = os.path.join(model_path, "res_mlp_pooler.pt")
            if not os.path.exists(state_path):
                logger.info("No res_mlp_pooler weights found at %s", state_path)
                return
            state = torch.load(state_path, map_location="cpu")
            missing, unexpected = self.res_mlp_pooler.load_state_dict(state, strict=False)
            logger.info(
                "Loaded res_mlp_pooler weights from %s (missing=%s, unexpected=%s)",
                state_path,
                len(missing),
                len(unexpected),
            )
        except Exception as e:
            logger.warning("Failed to load res_mlp_pooler weights: %s", e)
