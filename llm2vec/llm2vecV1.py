"这是此前所有实验所使用的版本"

import json
import logging
import os
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from peft import PeftModel
from torch import Tensor, device, nn
from tqdm.autonotebook import tqdm, trange
from transformers import (
    AutoModel,
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
)

from .models import (
    MistralBiModel,
    LlamaBiModel,
    GemmaBiModel,
    Qwen2BiModel,
)
from .pooling_latent import LatentAttentionPooling
from .pooling_structured_selfattn import StructuredSelfAttentionPooling

logger = logging.getLogger(__name__)


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class LLM2Vec(nn.Module):
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        pooling_mode: str = "mean",
        max_length: int = 512,
        doc_max_length: int = 400,
        skip_instruction: bool = True,
        selfattn_attn_hidden_dim: int = 512,
        selfattn_num_hops: int = 8,
        selfattn_output_dropout: float = 0.0,
        selfattn_output_layernorm: bool = True,
        selfattn_gamma_init: float = 1e-3,
        selfattn_gamma_learnable: bool = True,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        # Validate pooling_mode early to provide clear error messages
        valid_pooling_modes = {
            "mean",
            "weighted_mean",
            "eos_token",
            "last_token",
            "bos_token",
            "latent_pooling",
            "structured_selfattn",
        }
        if pooling_mode not in valid_pooling_modes:
            raise ValueError(
                f"Unsupported pooling_mode '{pooling_mode}'. Supported modes: {sorted(valid_pooling_modes)}"
            )
        self.pooling_mode = pooling_mode
        self.skip_instruction = skip_instruction
        self.max_length = max_length
        self.doc_max_length = doc_max_length
        self.config = model.config
        self.selfattn_attn_hidden_dim = selfattn_attn_hidden_dim
        self.selfattn_num_hops = selfattn_num_hops
        self.selfattn_output_dropout = selfattn_output_dropout
        self.selfattn_output_layernorm = selfattn_output_layernorm
        self.selfattn_gamma_init = selfattn_gamma_init
        self.selfattn_gamma_learnable = selfattn_gamma_learnable

        # Initialize latent attention pooling when requested
        self.latent_attn: Optional[LatentAttentionPooling] = None
        self.structured_self_attn: Optional[StructuredSelfAttentionPooling] = None
        self._pooling_aux_loss: Optional[Tensor] = None
        if self.pooling_mode == "latent_pooling":
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(
                    "Model config must define hidden_size to use latent_pooling."
                )
            # Default hyperparameters based on pooling_latent.py
            self.latent_attn = LatentAttentionPooling(
                d_model=hidden_size,
                num_latents=512,
                num_heads=8,
            )
        elif self.pooling_mode == "structured_selfattn":
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError(
                    "Model config must define hidden_size to use structured_selfattn."
                )
            self.structured_self_attn = StructuredSelfAttentionPooling(
                d_model=hidden_size,
                attn_hidden_dim=self.selfattn_attn_hidden_dim,
                num_hops=self.selfattn_num_hops,
                output_dropout=self.selfattn_output_dropout,
                output_layernorm=self.selfattn_output_layernorm,
                gamma_init=self.selfattn_gamma_init,
                gamma_learnable=self.selfattn_gamma_learnable,
            )
        else:
            self.latent_attn = None
            logger.info(f"Present pooling mode is {self.pooling_mode}. Latent attention pooling is not enabled.")

    def _infer_device(self) -> torch.device:
        """Infer a sensible device for the module even when it has no parameters.

        Tries `self.model`, then `self.latent_attn`, then `self` itself; falls back to
        CUDA if available, else CPU.
        """
        for m in (
            getattr(self, "model", None),
            getattr(self, "latent_attn", None),
            getattr(self, "structured_self_attn", None),
            self,
        ):
            if m is None:
                continue
            try:
                p = next(m.parameters())
                return p.device
            except StopIteration:
                continue
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def _get_model_class(cls, config_class_name, enable_bidirectional):
        if not enable_bidirectional:
            return AutoModel
        if config_class_name == "MistralConfig":
            return MistralBiModel
        elif config_class_name == "LlamaConfig":
            return LlamaBiModel
        elif config_class_name == "GemmaConfig":
            return GemmaBiModel
        elif config_class_name == "Qwen2Config":
            return Qwen2BiModel
        elif config_class_name == "Qwen3Config":
            return Qwen3BiModel
        else:
            raise ValueError(
                f"{config_class_name} is not supported yet with bidirectional models."
            )

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
        # pop out encoder args
        keys = [
            "pooling_mode",
            "max_length",
            "doc_max_length",
            "skip_instruction",
            "selfattn_attn_hidden_dim",
            "selfattn_num_hops",
            "selfattn_output_dropout",
            "selfattn_output_layernorm",
            "selfattn_gamma_init",
            "selfattn_gamma_learnable",
        ]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(base_model_name_or_path)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )
        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)
        logger.info(f"Loaded base model from {base_model_name_or_path}")
        
        if os.path.isdir(base_model_name_or_path) and os.path.exists(
            f"{base_model_name_or_path}/config.json"
        ):
            with open(f"{base_model_name_or_path}/config.json", "r") as fIn:
                config_dict = json.load(fIn)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        # For special case where config.json and adapter weights are in the same directory
        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
            )
            if merge_peft:
                model = model.merge_and_unload()
                logger.info("have merged PEFT adapter weights from %s", peft_model_name_or_path)
        else:
            logger.info("PEFT model is not provided.")

        # Optionally merge extra adapter checkpoints for composite models
        if extra_model_name_or_path is not None and len(extra_model_name_or_path) > 0:
            # Ensure base model is in non-PEFT form when chaining merges
            if not merge_peft and isinstance(model, PeftModel):
                model = model.merge_and_unload()
                logger.info("have merged base model weights from %s", base_model_name_or_path)
            if isinstance(extra_model_name_or_path, str):
                model = PeftModel.from_pretrained(
                    model,
                    extra_model_name_or_path,
                )
                model = model.merge_and_unload()
                logger.info("have merged extra PEFT adapter weights from %s", extra_model_name_or_path)
                # For downstream config address resolution
                peft_model_name_or_path = extra_model_name_or_path
            elif isinstance(extra_model_name_or_path, list):
                for extra_model in extra_model_name_or_path:
                    model = PeftModel.from_pretrained(
                        model,
                        extra_model,
                    )
                    model = model.merge_and_unload()
                    peft_model_name_or_path = extra_model
                    logger.info("have merged extra PEFT adapter weights from %s", extra_model)
            else:
                raise ValueError(
                    "extra_model_name_or_path should be a string or a list of strings."
                )
        else:
            logger.info("extra model is not provided.")

        config = {}
        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as fIn:
                llm2vec_config = json.load(fIn)
            config.update(llm2vec_config)

        for key, value in encoder_args.items():
            config[key] = value

        llm2vec_model = cls(model=model, tokenizer=tokenizer, **config)

        # Auto-load latent attention weights if using latent_pooling
        if (
            getattr(llm2vec_model, "latent_attn", None) is not None
            and llm2vec_model.pooling_mode == "latent_pooling"
        ):
            llm2vec_model._load_latent_attention_weights(
                peft_model_name_or_path, use_safetensors=use_safetensors
            )
        if (
            getattr(llm2vec_model, "structured_self_attn", None) is not None
            and llm2vec_model.pooling_mode == "structured_selfattn"
        ):
            llm2vec_model._load_structured_self_attention_weights(
                peft_model_name_or_path
            )

        # Ensure dtype conversion if requested
        if "torch_dtype" in kwargs and kwargs["torch_dtype"] is not None:
            llm2vec_model = llm2vec_model.to(kwargs["torch_dtype"])  # type: ignore

        return llm2vec_model

    def to(self, device_or_dtype):
        """
        Move the module to a device or dtype. Ensures latent attention pooling
        module is also moved appropriately.
        """
        result = super().to(device_or_dtype)
        if hasattr(result, "latent_attn") and result.latent_attn is not None:
            result.latent_attn = result.latent_attn.to(device_or_dtype)
        if hasattr(result, "structured_self_attn") and result.structured_self_attn is not None:
            result.structured_self_attn = result.structured_self_attn.to(device_or_dtype)
        return result

    def reset_pooling_aux_loss(self):
        self._pooling_aux_loss = None

    def _accumulate_pooling_aux_loss(self, aux_loss: Optional[Tensor]):
        if aux_loss is None:
            return
        if self._pooling_aux_loss is None:
            self._pooling_aux_loss = aux_loss
        else:
            self._pooling_aux_loss = self._pooling_aux_loss + aux_loss

    def consume_pooling_aux_loss(self, reset: bool = True) -> Optional[Tensor]:
        aux_loss = self._pooling_aux_loss
        if reset:
            self._pooling_aux_loss = None
        return aux_loss

    def _load_latent_attention_weights(self, peft_model_path: str, use_safetensors: bool = True):
        """
        Attempt to load latent attention weights from the given model path.
        Looks for files named 'latent_attn.pt' or 'latent_attn.safetensors'.

        This is best-effort: if weights are not found, the randomly initialized
        latent attention module will be used.
        """
        if self.latent_attn is None:
            return
        try:
            pt_path = os.path.join(peft_model_path, "latent_attn.pt")
            st_path = os.path.join(peft_model_path, "latent_attn.safetensors")

            def _try_load(state_dict: dict):
                """Load state dict, accommodating possible 'latent_attn.' prefixes."""
                # If keys are prefixed with 'latent_attn.', strip them
                if any(k.startswith("latent_attn.") for k in state_dict.keys()):
                    state_dict = {k.replace("latent_attn.", ""): v for k, v in state_dict.items()}
                missing, unexpected = self.latent_attn.load_state_dict(state_dict, strict=False)
                logger.info(
                    f"Loaded latent_attn weights (missing={len(missing)}, unexpected={len(unexpected)})"
                )

            if use_safetensors and os.path.exists(st_path):
                try:
                    from safetensors.torch import load_file  # type: ignore
                    state = load_file(st_path)
                    _try_load(state)
                    logger.info(f"Loaded latent_attn weights from {st_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load safetensors weights: {e}. Falling back to .pt if available.")
            if os.path.exists(pt_path):
                state = torch.load(pt_path, map_location="cpu")
                _try_load(state)
                logger.info(f"Loaded latent_attn weights from {pt_path}")
        except Exception as e:
            logger.warning(f"="*15 + "Training latent attention weights from scratch" + "="*15)
            logger.warning(f"there is no latent_attn weights in {peft_model_path} or {peft_model_path} is not provided.")

    def _load_structured_self_attention_weights(self, model_path: Optional[str]):
        if self.structured_self_attn is None or model_path is None:
            return
        try:
            state_path = os.path.join(model_path, "structured_self_attn.pt")
            if not os.path.exists(state_path):
                logger.info("No structured self-attn weights found at %s", state_path)
                return
            state = torch.load(state_path, map_location="cpu")
            if any(k.startswith("structured_self_attn.") for k in state.keys()):
                state = {
                    k.replace("structured_self_attn.", ""): v for k, v in state.items()
                }
            # Backward compatibility: older checkpoints may not save gamma.
            if "gamma" not in state and hasattr(self.structured_self_attn, "gamma"):
                current_state = self.structured_self_attn.state_dict()
                if "gamma" in current_state:
                    state["gamma"] = current_state["gamma"].detach().clone()
                    logger.info(
                        "structured_self_attn checkpoint at %s does not contain gamma; "
                        "using the current module default value instead.",
                        state_path,
                    )
            missing, unexpected = self.structured_self_attn.load_state_dict(
                state, strict=False
            )
            logger.info(
                "Loaded structured self-attn weights from %s (missing=%s, unexpected=%s)",
                state_path,
                len(missing),
                len(unexpected),
            )
        except Exception as e:
            logger.warning("Failed to load structured self-attn weights: %s", e)

    def prepare_for_tokenization(self, text):
        if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct" or isinstance(self.model.config, LlamaConfig):
            text = (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                + text.strip()
                + "<|eot_id|>"
            )
            return text
        if self.model.config._name_or_path in [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
        ]:
            text = "[INST] " + text.strip() + " [/INST]"
        if self.model.config._name_or_path in [
            "google/gemma-2-9b-it",
        ]:
            text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
        if self.model.config._name_or_path in [
            "Qwen/Qwen2-1.5B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
        ]:
            text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
        if self.pooling_mode == "eos_token":
            if self.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
                text = text.strip() + "<|end_of_text|>"
            elif isinstance(self.model.config, LlamaConfig) or isinstance(
                self.model.config, MistralConfig
            ):
                text = text.strip() + " </s>"
            elif isinstance(self.model.config, GemmaConfig):
                text = text.strip() + "<eos>"
            elif isinstance(self.model.config, Qwen2Config):
                text = text.strip() + "<|endoftext|>"
        return text

    def tokenize(self, texts):
        texts_2 = []
        original_texts = []
        for text in texts:
            t = text.split("!@#$%^&*()")
            texts_2.append(t[1] if len(t) > 1 else "")
            original_texts.append("".join(t))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            if embed_mask is None:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = e_m.unsqueeze(0)
            else:
                e_m = torch.zeros_like(original["attention_mask"][t_i])
                if len(ids["input_ids"][0]) > 0:
                    e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                        len(ids["input_ids"][0])
                    )
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original

    def _skip_instruction(self, sentence_feature):
        # Use embed_mask if provided; otherwise fall back to attention_mask.
        # 记录实际使用的掩码类型
        mask_type = "embed_mask" if "embed_mask" in sentence_feature else "attention_mask"
        logger.debug(f"_skip_instruction 使用 {mask_type} 作为掩码")
        embed_mask = sentence_feature.get("embed_mask", sentence_feature.get("attention_mask"))
        assert embed_mask is not None, "Missing both 'embed_mask' and 'attention_mask' for skip_instruction."
        # Ensure shapes align with attention_mask when present.
        if "attention_mask" in sentence_feature:
            assert (
                sentence_feature["attention_mask"].shape == embed_mask.shape
            ), "embed_mask and attention_mask must have the same shape"
        sentence_feature["attention_mask"] = embed_mask

    def get_pooling(self, features, last_hidden_states):  # All models padded from left
        assert (
            self.tokenizer.padding_side == "left"
        ), "Pooling modes are implemented for padding from left."
        if self.skip_instruction:
            self._skip_instruction(features)
        seq_lengths = features["attention_mask"].sum(dim=-1)

        # Check for zero-length sequences
        if (seq_lengths == 0).any():
            raise ValueError(
                "All attention masks are zero. This may be caused by missing separator in text data. "
                "Please ensure input text format is correct and contains the separator '!@#$%^&*()'."
            )

        if self.pooling_mode == "mean":
            return torch.stack(
                [
                    last_hidden_states[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        elif self.pooling_mode == "weighted_mean":
            bs, l, _ = last_hidden_states.shape
            complete_weights = torch.zeros(bs, l, device=last_hidden_states.device)
            for i, seq_l in enumerate(seq_lengths):
                if seq_l > 0:
                    complete_weights[i, -seq_l:] = torch.arange(seq_l) + 1
                    complete_weights[i] /= torch.clamp(
                        complete_weights[i].sum(), min=1e-9
                    )
            return torch.sum(last_hidden_states * complete_weights.unsqueeze(-1), dim=1)
        elif self.pooling_mode == "eos_token" or self.pooling_mode == "last_token":
            return last_hidden_states[:, -1]
        elif self.pooling_mode == "bos_token":
            return last_hidden_states[
                features["input_ids"] == self.tokenizer.bos_token_id
            ]
        elif self.pooling_mode == "latent_pooling":
            if self.latent_attn is None:
                raise RuntimeError(
                    "latent_attn module is not initialized but latent_pooling was selected."
                )
            # Prefer embed_mask (instruction-aware) if present, otherwise use attention_mask
            attn_mask = None
            if "embed_mask" in features and features["embed_mask"] is not None:
                attn_mask = features["embed_mask"].to(last_hidden_states.device)
            else:
                attn_mask = features.get("attention_mask", None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(last_hidden_states.device)
            return self.latent_attn(last_hidden_states, attention_mask=attn_mask)
        elif self.pooling_mode == "structured_selfattn":
            if self.structured_self_attn is None:
                raise RuntimeError(
                    "structured_self_attn module is not initialized but structured_selfattn was selected."
                )
            attn_mask = None
            if "embed_mask" in features and features["embed_mask"] is not None:
                attn_mask = features["embed_mask"].to(last_hidden_states.device)
            else:
                attn_mask = features.get("attention_mask", None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(last_hidden_states.device)
            pooled, aux_loss = self.structured_self_attn(
                last_hidden_states,
                attention_mask=attn_mask,
            )
            self._accumulate_pooling_aux_loss(aux_loss)
            return pooled
        else:
            raise ValueError(f"{self.pooling_mode} is not implemented yet.")

    def forward(self, sentence_feature: Dict[str, Tensor]):
        embed_mask = None
        if "embed_mask" in sentence_feature:
            embed_mask = sentence_feature.pop("embed_mask")
        reps = self.model(**sentence_feature)
        sentence_feature["embed_mask"] = embed_mask

        return self.get_pooling(sentence_feature, reps.last_hidden_state)



    def _convert_to_str(self, instruction, text):
        tokenized_q = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        tokenized_q_length = len(tokenized_q["input_ids"][0])

        while tokenized_q_length > self.doc_max_length:
            reduction_ratio = self.doc_max_length / tokenized_q_length
            reduced_length = int(len(text.split()) * reduction_ratio)
            text = " ".join(text.split()[:reduced_length])
            tokenized_q = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            tokenized_q_length = len(tokenized_q["input_ids"][0])

        return (
            f"{instruction.strip()} !@#$%^&*(){text}"
            if instruction
            else f"!@#$%^&*(){text}"
        )

    # ---------------------------------------------------------------------
    # High-level convenience APIs (migrated from llm2vec_wrapper.py)
    # ---------------------------------------------------------------------
    def encode_text(self, text: Union[str, List[str]], max_length: Optional[int] = None) -> Tensor:
        """
        Encode plain text or list of texts (without instruction or separator) to embeddings, automatically handling `embed_mask`.

        Args:
            text: A single string or a list of strings to encode.
            max_length: Optional maximum sequence length; defaults to `self.max_length`.

        Returns:
            Torch tensor of shape `(batch, hidden_size)` containing embeddings.
        """
        if max_length is None:
            max_length = getattr(self, "max_length", 512)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # For simple text encoding, use attention_mask as embed_mask
        inputs["embed_mask"] = inputs["attention_mask"].clone()

        model_device = self._infer_device()
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self(inputs)

        return embeddings

    def tokenize_with_separator(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        separator: str = "!@#$%^&*()",
    ) -> Dict[str, Tensor]:
        """
        Tokenize texts with a separator that splits instruction and content.

        This mirrors the wrapper behavior while aligning with internal tokenize logic.

        Args:
            texts: List of input strings potentially containing a separator.
            max_length: Optional maximum sequence length; defaults to `self.max_length`.
            separator: Separator string used to split instruction from text.

        Returns:
            Dict with `input_ids`, `attention_mask`, and `embed_mask` tensors.
        """
        if max_length is None:
            max_length = getattr(self, "max_length", 512)

        texts_2: List[str] = []
        original_texts: List[str] = []

        for text in texts:
            parts = text.split(separator)
            texts_2.append(parts[1] if len(parts) > 1 else "")
            original_texts.append("".join(parts))

        original = self.tokenizer(
            original_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        embed_mask = None
        for t_i, t in enumerate(texts_2):
            ids = self.tokenizer(
                [t],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )

            e_m = torch.zeros_like(original["attention_mask"][t_i])
            if len(ids["input_ids"][0]) > 0:
                e_m[-len(ids["input_ids"][0]) :] = torch.ones(len(ids["input_ids"][0]))

            if embed_mask is None:
                embed_mask = e_m.unsqueeze(0)
            else:
                embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

        original["embed_mask"] = embed_mask
        return original

    def encode_with_instruction(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
        separator: str = "!@#$%^&*()",
        show_progress_bar: bool = False,
        convert_to_numpy: bool = False,
        device: Optional[str] = None,
    ) -> Tensor:
        """
        Encode texts prepared for "instruction-separator-text" pairs via a separator.
        Supports batch processing and multi-GPU acceleration.

        Args:
            texts: List of texts containing instruction and content separated by `separator`.
            batch_size: Batch size for encoding.
            max_length: Optional maximum sequence length.
            separator: Separator string.
            show_progress_bar: Whether to show progress bar.
            convert_to_numpy: If true, return numpy arrays instead of torch tensors.
            device: Target device (e.g., 'cuda', 'cpu'). If None, auto-detect.

        Returns:
            Torch tensor of shape `(batch, hidden_size)` containing embeddings.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if max_length is None:
            max_length = getattr(self, "max_length", 512)

        self.eval()
        all_embeddings = []

        if torch.cuda.device_count() <= 1:
            # Single GPU or CPU
            self.to(device)
            for start_index in trange(
                0,
                len(texts),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                texts_batch = texts[start_index : start_index + batch_size]
                embeddings = self._encode_with_separator(
                    texts_batch,
                    device=device,
                    max_length=max_length,
                    separator=separator,
                    convert_to_numpy=convert_to_numpy,
                )
                all_embeddings.append(embeddings)
        else:
            # Multi-GPU processing
            num_proc = torch.cuda.device_count()
            cuda_compatible_multiprocess = mp.get_context("spawn")
            with cuda_compatible_multiprocess.Pool(num_proc) as p:
                texts_batches = [
                    texts[start_index : start_index + batch_size]
                    for start_index in range(0, len(texts), batch_size)
                ]

                progress_bar = tqdm(
                    total=len(texts_batches),
                    desc="Batches",
                    disable=not show_progress_bar,
                )
                results = []

                def update(*args):
                    progress_bar.update()

                for batch in texts_batches:
                    results.append(
                        p.apply_async(
                            self._encode_with_separator,
                            args=(batch, None, max_length, separator, convert_to_numpy, True),
                            callback=update,
                        )
                    )

                all_embeddings = [result.get() for result in results]
                progress_bar.close()

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings.to(torch.float32)
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def encode_with_separator(
        self,
        texts: List[str],
        device: Optional[Union[str, torch.device]] = None,
        max_length: Optional[int] = None,
        separator: str = "!@#$%^&*()",
    ) -> Tensor:
        """
        Convenience method to encode texts that include instruction/content pairs. device can be specified.

        Args:
            texts: List of input strings potentially containing a separator.
            device: Target device; defaults to current module device.
            max_length: Maximum sequence length; defaults to `self.max_length`.
            separator: Separator string used to split instruction from text.

        Returns:
            Torch tensor embeddings of shape `(batch, hidden_size)`.
        """
        if device is None:
            device = self._infer_device()
        if max_length is None:
            max_length = getattr(self, "max_length", 512)

        self.to(device)

        tokenized = self.tokenize_with_separator(texts, max_length=max_length, separator=separator)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        tokenized = {
            k: (v.to(self.model.dtype) if v.dtype.is_floating_point else v) for k, v in tokenized.items()
        }

        with torch.no_grad():
            embeddings = self(tokenized)

        return embeddings

    def compute_similarities(
        self,
        query_text: str,
        candidate_texts: List[str],
        device: Optional[Union[str, torch.device]] = None,
        separator: str = "!@#$%^&*()",
    ) -> Tensor:
        """
        Compute cosine similarity between a query text and candidate texts.

        Args:
            query_text: Query string, optionally using the separator for instruction/content.
            candidate_texts: List of candidate strings.
            device: Target device; defaults to module device.
            separator: Separator string for instruction/content pairs.

        Returns:
            1D torch.Tensor of similarity scores, one per candidate.
        """
        import torch.nn.functional as F

        if device is None:
            device = self._infer_device()

        all_texts = [query_text] + candidate_texts
        embeddings = self.encode_with_separator(all_texts, device=device, separator=separator)

        # embeddings shape: (batch, hidden); compare first with rest
        sims = F.cosine_similarity(embeddings[0], embeddings[1:], dim=1)
        return sims

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = False,
        convert_to_tensor: bool = False,
        device: Optional[str] = None,
    ):
        """
        Encode a list of sentences to their respective embeddings. The sentences can be a list of strings or a string.
        Args:
            sentences: List[str]:["Hello world", "This is a test"] or List[List[str]] or List[Tuple[str, str]]:[[instruction, text], ["Translate this:", "Hello world"], ["Summarize:", "Long text..."]]
            batch_size: batch size for turning sentence tokens into embeddings.
            show_progress_bar: whether to show progress bars during encoding steps.
            convert_to_numpy: If true, return numpy arrays instead of torch tensors.
            convert_to_tensor: If true, return torch tensors (default).
            device: torch backend device identifier (e.g., 'cuda', 'cpu','mps' etc.). If not specified,
            the default is to use cuda when available, otherwise cpu. Note that only the choice of 'cuda' supports
            multiprocessing as currently implemented.

        Returns: embeddings of the sentences. Embeddings are detached and always on the CPU (see _encode implementation).

        """
        if isinstance(sentences[0], str) and isinstance(sentences[-1], int):
            sentences = [sentences]
        # required for MEDI version of MTEB
        if isinstance(sentences[0], str):
            sentences = [[""] + [sentence] for sentence in sentences]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        concatenated_input_texts = []
        for sentence in sentences:
            assert isinstance(sentence[0], str)
            assert isinstance(sentence[1], str)
            concatenated_input_texts.append(
                self._convert_to_str(sentence[0], sentence[1])
            )
        sentences = concatenated_input_texts

        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        all_embeddings = []

        if torch.cuda.device_count() <= 1:
            # This branch also support mps devices
            self.to(device)
            for start_index in trange(
                0,
                len(sentences),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                sentences_batch = sentences_sorted[
                    start_index : start_index + batch_size
                ]
                embeddings = self._encode(
                    sentences_batch, device=device, convert_to_numpy=convert_to_numpy
                )
                all_embeddings.append(embeddings)
        else:
            num_proc = torch.cuda.device_count()
            cuda_compatible_multiprocess = mp.get_context("spawn")
            with cuda_compatible_multiprocess.Pool(num_proc) as p:
                sentences_batches = [
                    sentences_sorted[start_index : start_index + batch_size]
                    for start_index in range(0, len(sentences), batch_size)
                ]

                progress_bar = tqdm(
                    total=len(sentences_batches),
                    desc="Batches",
                    disable=not show_progress_bar,
                )
                results = []

                def update(*args):
                    progress_bar.update()

                for batch in sentences_batches:
                    results.append(
                        p.apply_async(
                            self._encode,
                            args=(batch, None, convert_to_numpy, True),
                            callback=update,
                        )
                    )

                all_embeddings = [result.get() for result in results]
                progress_bar.close()

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        all_embeddings = all_embeddings.to(torch.float32)
        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings

    def save(self, output_path, merge_before_save=False, save_config=True):
        if merge_before_save and isinstance(self.model, PeftModel):
            self.model = self.model.merge_and_unload()
            # Fixes the issue of saving - https://huggingface.co/McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse/discussions/1
            if hasattr(self.model, "_hf_peft_config_loaded"):
                self.model._hf_peft_config_loaded = False

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        llm2vec_config = {
            "pooling_mode": self.pooling_mode,
            "max_length": self.max_length,
            "doc_max_length": self.doc_max_length,
            "skip_instruction": self.skip_instruction,
            "selfattn_attn_hidden_dim": self.selfattn_attn_hidden_dim,
            "selfattn_num_hops": self.selfattn_num_hops,
            "selfattn_output_dropout": self.selfattn_output_dropout,
            "selfattn_output_layernorm": self.selfattn_output_layernorm,
            "selfattn_gamma_init": self.selfattn_gamma_init,
            "selfattn_gamma_learnable": self.selfattn_gamma_learnable,
        }

        if save_config:
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/llm2vec_config.json", "w") as fOut:
                json.dump(llm2vec_config, fOut, indent=4)

        # Optionally save latent attention weights alongside the encoder
        if getattr(self, "latent_attn", None) is not None:
            try:
                torch.save(self.latent_attn.state_dict(), os.path.join(output_path, "latent_attn.pt"))
            except Exception as e:
                logger.warning(f"Failed to save latent_attn weights: {e}")
        if getattr(self, "structured_self_attn", None) is not None:
            try:
                torch.save(
                    self.structured_self_attn.state_dict(),
                    os.path.join(output_path, "structured_self_attn.pt"),
                )
            except Exception as e:
                logger.warning(f"Failed to save structured_self_attn weights: {e}")

    def _encode(
        self,
        sentences_batch,
        device: Optional[str] = None,
        convert_to_numpy: bool = False,
        multiprocessing=False,
    ):
        if multiprocessing:
            # multiprocessing only supports CUDA devices at this time, so we ignore the value of device
            # and use cuda:rank for the device
            rank = mp.current_process()._identity[0]
            if device is None and torch.cuda.is_available():
                device = f"cuda:{rank % torch.cuda.device_count()}"

        self.to(device)
        features = self.tokenize(
            [self.prepare_for_tokenization(sentence) for sentence in sentences_batch]
        )
        features = batch_to_device(features, device)

        with torch.no_grad():
            embeddings = self.forward(features)
            embeddings = embeddings.detach()
            embeddings = embeddings.cpu()

        return embeddings

    def _encode_with_separator(
        self,
        texts_batch: List[str],
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        separator: str = "!@#$%^&*()",
        convert_to_numpy: bool = False,
        multiprocessing: bool = False,
    ):
        """
        Helper method to encode a batch of texts with separator for multiprocessing support.
        """
        if multiprocessing:
            rank = mp.current_process()._identity[0]
            if device is None and torch.cuda.is_available():
                device = f"cuda:{rank % torch.cuda.device_count()}"

        if device is None:
            device = self._infer_device()
        if max_length is None:
            max_length = getattr(self, "max_length", 512)

        self.to(device)
        tokenized = self.tokenize_with_separator(texts_batch, max_length=max_length, separator=separator)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        tokenized = {
            k: (v.to(self.model.dtype) if v.dtype.is_floating_point else v) for k, v in tokenized.items()
        }

        with torch.no_grad():
            embeddings = self(tokenized)
            embeddings = embeddings.detach()
            embeddings = embeddings.cpu()

        return embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either a string (which means a single text)
        a list of ints (which means a single tokenized text), or a tuple of list of ints
        (representing several text inputs to the model).
        """
        if (
            isinstance(text, str)
            or (isinstance(text, list) and isinstance(text[0], int))
            or len(text) == 0
        ):  # Single text, list of ints, or empty
            return len(text)
        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        else:
            return sum([len(t) for t in text])

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        return self.model.resize_token_embeddings(
            new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def convert_to_bert_format(
        self,
        sentence_feature: Dict[str, Tensor],
        pooling: Optional[str] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, Tensor]:
        """
        Convert model outputs to a BERT-compatible dictionary format.

        Returns a dict with keys:
        - 'last_hidden_state': (batch, seq_len, hidden_size) float32 tensor
        - 'pooler_output': (batch, hidden_size) float32 tensor computed by selected pooling

        Args:
            sentence_feature: Input feature dict for the encoder model.
            pooling: Override pooling mode for pooler_output. If None, uses self.pooling_mode.
            return_tensors: Currently supports 'pt' only; reserved for future numpy support.
        """
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported for convert_to_bert_format().")

        embed_mask = sentence_feature.get("embed_mask", None)
        reps = self.model(**{k: v for k, v in sentence_feature.items() if k != "embed_mask"})
        last_hidden = reps.last_hidden_state.to(torch.float32)

        # Temporarily inject embed_mask for pooling
        tmp_features = dict(sentence_feature)
        tmp_features["embed_mask"] = embed_mask

        selected_pooling = pooling if pooling is not None else self.pooling_mode
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
