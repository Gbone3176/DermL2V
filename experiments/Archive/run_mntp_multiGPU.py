#!/usr/bin/env python
# coding=utf-8

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Any, Tuple, List

import numpy as np
import datasets
import evaluate
from datasets import load_dataset

import torch
import torch.distributed as dist
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from peft import LoraConfig, get_peft_model, PeftModel

from llm2vec.models import (
    MistralBiForMNTP,
    LlamaBiForMNTP,
    GemmaBiForMNTP,
    Qwen2BiForMNTP,
)

from swanlab.integration.transformers import SwanLabCallback

require_version("datasets>=1.8.0", "To fix: pip install datasets>=1.8.0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# ---------------------------
# DDP helpers (critical fix)
# ---------------------------
def ddp_env():
    """Return (is_ddp, local_rank, rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        return True, int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, -1, -1, 1


def ddp_init_or_fix():
    """
    Ensure:
      1) CUDA device is set to LOCAL_RANK
      2) ProcessGroup is initialized with device_id to avoid barrier() guessing device
    """
    is_ddp, local_rank, rank, world_size = ddp_env()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not is_ddp:
        return

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA, but torch.cuda.is_available() is False.")

    torch.cuda.set_device(local_rank)
    # force CUDA context on correct device
    _ = torch.empty(1, device=f"cuda:{local_rank}")

    # If someone (HF / accelerate) already initialized process group without device_id, destroy and re-init.
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    # Init with device_id to stop PyTorch from guessing device in barrier()
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )


def ddp_barrier():
    """Barrier with explicit device_ids to avoid guessing."""
    if dist.is_available() and dist.is_initialized():
        is_ddp, local_rank, _, _ = ddp_env()
        if is_ddp and local_rank >= 0:
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()


def is_main_process() -> bool:
    is_ddp, _, rank, _ = ddp_env()
    return (not is_ddp) or (rank == 0)


# ---------------------------
# model utils
# ---------------------------
def get_model_class(config):
    config_class_name = config.__class__.__name__
    if config_class_name == "MistralConfig":
        return MistralBiForMNTP
    elif config_class_name == "LlamaConfig":
        return LlamaBiForMNTP
    elif config_class_name == "GemmaConfig":
        return GemmaBiForMNTP
    elif config_class_name == "Qwen2Config":
        return Qwen2BiForMNTP
    else:
        raise ValueError(f"Model class {config_class_name} not supported.")


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, config)
    print("Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


# ---------------------------
# args
# ---------------------------
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    token: str = field(default=None)
    use_auth_token: bool = field(default=None)
    trust_remote_code: bool = field(default=False)
    torch_dtype: Optional[str] = field(default=None, metadata={"choices": ["auto", "bfloat16", "float16", "float32"]})
    attn_implementation: Optional[str] = field(default="sdpa", metadata={"choices": ["eager", "sdpa", "flash_attention_2"]})
    low_cpu_mem_usage: bool = field(default=False)
    peft_model_name_or_path: Optional[str] = field(default=None)
    extra_model_name_or_path: List[str] = field(default_factory=list)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=True)
    validation_split_percentage: Optional[int] = field(default=5)
    max_seq_length: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)  # 建议=1/None
    mlm_probability: float = field(default=0.15)
    line_by_line: bool = field(default=False)
    pad_to_max_length: bool = field(default=False)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    streaming: bool = field(default=False)


@dataclass
class CustomArguments:
    lora_dropout: float = field(default=0.05)
    lora_r: int = field(default=8)
    mask_token_type: str = field(default="blank")
    stop_after_n_steps: int = field(default=10000)
    data_collator_type: str = field(default="default")


# ---------------------------
# collator / trainer
# ---------------------------
class DataCollatorForLanguageModelingWithFullMasking(DataCollatorForLanguageModeling):
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class MNTPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["labels"]

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        return dataset

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_peft_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


# ---------------------------
# main
# ---------------------------
def main():
    # Parse first
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    # Critical: fix DDP *before any barrier/map*
    ddp_init_or_fix()

    # gradient checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # auth token compat
    if model_args.use_auth_token is not None:
        warnings.warn("use_auth_token is deprecated; use token instead.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("Both token and use_auth_token are specified.")
        model_args.token = model_args.use_auth_token

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) exists and is not empty. "
                "Use --overwrite_output_dir."
            )

    set_seed(training_args.seed)

    # Load dataset
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
    else:
        data_files = {}
        extension = None
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # Ensure validation split exists
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            "text" if data_args.train_file and data_args.train_file.endswith(".txt") else extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        raw_datasets["train"] = load_dataset(
            "text" if data_args.train_file and data_args.train_file.endswith(".txt") else extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

    # Config + tokenizer
    config_kwargs = dict(
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = dict(
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    if tokenizer.mask_token is None:
        if custom_args.mask_token_type == "blank":
            tokenizer.mask_token = "_"
        elif custom_args.mask_token_type == "eos":
            tokenizer.mask_token = tokenizer.eos_token
        elif custom_args.mask_token_type == "mask":
            tokenizer.add_tokens(["<mask>"])
            tokenizer.mask_token = "<mask>"
        else:
            raise ValueError(f"mask_token_type {custom_args.mask_token_type} not supported")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # max seq length
    if data_args.max_seq_length is None:
        max_seq_length = min(tokenizer.model_max_length, 1024)
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # preprocess
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        return {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }

    # IMPORTANT: only rank0 does heavy map; others wait then read cache
    if is_main_process():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=None,  # 强制单进程，最稳
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=None,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )
    ddp_barrier()

    if not is_main_process():
        # 其他 rank 直接走一遍 map，但会从 cache 读取（几乎不耗时）
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=None,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Loading tokenized cache (rank>0)",
        )
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=None,
            load_from_cache_file=True,
            desc="Loading grouped cache (rank>0)",
        )
    ddp_barrier()

    train_dataset = tokenized_datasets["train"] if training_args.do_train else None
    eval_dataset = tokenized_datasets["validation"] if training_args.do_eval else None

    if training_args.do_train and data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
    if training_args.do_eval and data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    # Load model
    model_class = get_model_class(config)
    torch_dtype = model_args.torch_dtype
    torch_dtype = torch_dtype if torch_dtype in ["auto", None] else getattr(torch, torch_dtype)

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation,
    )

    # merge adapters if any
    if model_args.peft_model_name_or_path:
        peft_loaded = PeftModel.from_pretrained(model.model, model_args.peft_model_name_or_path)
        try:
            model.model = peft_loaded.merge_and_unload()
        except Exception:
            model.model = peft_loaded

    for extra in (model_args.extra_model_name_or_path or []):
        model.model = PeftModel.from_pretrained(model.model, extra)
        model.model = model.model.merge_and_unload()

    # resize embeddings BEFORE LoRA
    emb_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > emb_size:
        model.resize_token_embeddings(len(tokenizer))

    # apply LoRA
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    # metrics (eval)
    if training_args.do_eval:
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            preds = preds[:, :-1]
            labels = labels[:, 1:]
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            return metric.compute(predictions=preds[mask], references=labels[mask])
    else:
        compute_metrics = None
        preprocess_logits_for_metrics = None

    # collator
    if custom_args.data_collator_type == "all_mask":
        collator_cls = DataCollatorForLanguageModelingWithFullMasking
    elif custom_args.data_collator_type == "default":
        collator_cls = DataCollatorForLanguageModeling
    else:
        raise ValueError(f"data_collator_type {custom_args.data_collator_type} not supported")

    data_collator = collator_cls(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8,
    )

    swanlab_callback = SwanLabCallback(
        experiment_name="llm2vec-mntp-based-on-LLM2VEC-8B",
        description="MNTP pretrain with LoRA on Derm1M",
        tags=["MNTP", "LLM2Vec", "Derm1M"],
    )

    trainer = MNTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None),
        preprocess_logits_for_metrics=(
            preprocess_logits_for_metrics if training_args.do_eval and not is_torch_tpu_available() else None
        ),
    )
    trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))
    try:
        trainer.add_callback(swanlab_callback)
    except Exception as e:
        logger.warning(f"Failed to add SwanLab callback: {e}")

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except OverflowError:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
