import json
import os
from typing import List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer


def resolve_dtype(fp16: bool) -> torch.dtype:
    return torch.float16 if fp16 else torch.float32


def load_nvembed_model(model_name_or_path: str, fp16: bool, local_files_only: bool):
    dtype = resolve_dtype(fp16)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def enable_gradient_checkpointing(module: torch.nn.Module) -> None:
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable()
    if hasattr(module, "enable_input_require_grads"):
        module.enable_input_require_grads()
        return

    input_embeddings = module.get_input_embeddings()

    def _make_inputs_require_grad(_, __, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(_make_inputs_require_grad)


def attach_lora_to_text_encoder(
    model,
    target_modules: List[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    freeze_module(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )
    model.embedding_model = get_peft_model(model.embedding_model, config)
    return model


def trainable_parameter_summary(model: torch.nn.Module) -> str:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    ratio = 100.0 * trainable / max(total, 1)
    return f"trainable={trainable:,} total={total:,} ratio={ratio:.4f}%"


def move_batch_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(item, device) for item in batch]
    return batch


def encode_batch(model, batch_inputs):
    outputs = model(**batch_inputs)
    return outputs["sentence_embeddings"]


def save_training_artifacts(model, tokenizer, output_dir: str, config_dict: dict, state_dict: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, "adapter")
    model.embedding_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "training_state.json"), "w", encoding="utf-8") as handle:
        json.dump(state_dict, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "base_model_path.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(config_dict.get("model_name_or_path", "")) + "\n")
    with open(os.path.join(output_dir, "adapter_scope.txt"), "w", encoding="utf-8") as handle:
        handle.write("LoRA is attached to model.embedding_model only.\n")
