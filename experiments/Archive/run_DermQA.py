import logging
from dataclasses import dataclass, field
import os
import sys
import csv
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import seed_worker

from peft import LoraConfig, get_peft_model

from llm2vec import LLM2Vec
from llm2vec.experiment_utils import generate_experiment_id

transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")


def prepare_for_tokenization(model, text, pooling_mode="mean"):
    from transformers import LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config
    if getattr(model.config, "_name_or_path", None) == "meta-llama/Meta-Llama-3-8B-Instruct" or isinstance(model.config, LlamaConfig):
        text = "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        return text
    # if getattr(model.config, "_name_or_path", None) in [
    #     "mistralai/Mistral-7B-Instruct-v0.2",
    #     "meta-llama/Llama-2-7b-chat-hf",
    # ]:
    #     text = "[INST] " + text.strip() + " [/INST]"
    # if getattr(model.config, "_name_or_path", None) in [
    #     "google/gemma-2-9b-it",
    # ]:
    #     text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    # if getattr(model.config, "_name_or_path", None) in [
    #     "Qwen/Qwen2-1.5B-Instruct",
    #     "Qwen/Qwen2-7B-Instruct",
    # ]:
    #     text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if getattr(model.config, "_name_or_path", None) == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(model.config, MistralConfig):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
    return text


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and (
        model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]
    ):
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
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
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    peft_model_name_or_path: Optional[str] = field(default=None)
    bidirectional: Optional[bool] = field(default=True)
    max_seq_length: Optional[int] = field(default=512)
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"choices": ["auto", "bfloat16", "float16", "float32"]})
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"choices": ["eager", "sdpa", "flash_attention_2"]})
    pooling_mode: Optional[str] = field(default="mean", metadata={"choices": ["mean", "weighted_mean", "eos_token"]})


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    data_file: Optional[str] = field(default=None)
    val_ratio: float = field(default=0.1)
    test_ratio: float = field(default=0.1)
    separator: str = field(default="!@#$%^&*()")


@dataclass
class CustomArguments:
    lora_dropout: float = field(default=0.05)
    lora_r: int = field(default=8)
    stop_after_n_steps: int = field(default=None)
    experiment_id: Optional[str] = field(default=None)
    loss_class: Optional[str] = field(default="HardNegativeNLLLoss")
    loss_scale: float = field(default=50.0)
    recall_ks: Optional[List[int]] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class TrainSample:
    guid: str
    texts: List[str]
    label: Union[int, float]


class QACsvDataset(Dataset):
    def __init__(self, file_path: Optional[str] = None, rows: Optional[List[Dict[str, str]]] = None, split: str = "train", separator: str = "!@#$%^&*()"):
        self.split = split
        self.separator = separator
        if rows is None:
            self.rows = self._read_csv(file_path)
        else:
            self.rows = rows
        self.instruction = "Given a question related to skin diseases, retrieve the most relevant and appropriate answer to that question"

    def _read_csv(self, file_path: str) -> List[Dict[str, str]]:
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "prompt" in r and "response" in r:
                    rows.append({"prompt": r["prompt"], "response": r["response"]})
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return TrainSample(guid=str(idx), texts=[f"{self.instruction}{self.separator}{r['prompt']}", f"{self.instruction}{self.separator}{r['response']}"], label=1.0)


@dataclass
class DefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, features: List[TrainSample]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []
        for example in batch:
            for idx, text in enumerate(example.texts):
                text = prepare_for_tokenization(self.model, text, pooling_mode=self.model.pooling_mode)
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)
        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)
        return sentence_features, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps
    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_after_n_steps is not None and state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class LLM2VecSupervisedTrainer(Trainer):
    def __init__(self, *args, loss_function=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])
        d_reps_neg = None
        if len(features) > 2:
            d_reps_neg = self.model(features[2])
        loss = self.loss_function(q_reps, d_reps, d_reps_neg)
        if return_outputs:
            output = torch.stack([q_reps, d_reps], dim=1)
            return loss, output
        return loss
    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        features, labels = inputs
        with torch.no_grad():
            q_reps = self.model(features[0])
            d_reps = self.model(features[1])
            d_reps_neg = None
            if len(features) > 2:
                d_reps_neg = self.model(features[2])
            loss = None
            if not prediction_loss_only and self.loss_function is not None:
                loss = self.loss_function(q_reps, d_reps, d_reps_neg)
            logits = torch.stack([q_reps, d_reps], dim=1)
            if prediction_loss_only:
                return (loss if loss is not None else torch.tensor(0.0, device=logits.device), None, None)
            return (loss, logits, labels)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def build_compute_metrics(ks: List[int]):
    ks = sorted(set(int(k) for k in ks if k is not None and k > 0))
    def _compute(p):
        preds = p.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        q = preds[:, 0, :]
        d = preds[:, 1, :]
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        sims = np.matmul(qn, dn.T)
        ranks = np.argsort(-sims, axis=1)
        idx = np.arange(q.shape[0])
        results = {}
        for k in ks:
            hits_k = np.any(ranks[:, :k] == idx[:, None], axis=1)
            results[f"recall@{k}"] = float(np.mean(hits_k))
        return results
    return _compute


def split_rows(rows: List[Dict[str, str]], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n = len(rows)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_idxs = set(idxs[:n_val])
    test_idxs = set(idxs[n_val:n_val + n_test])
    train_rows, val_rows, test_rows = [], [], []
    for i in range(n):
        if i in val_idxs:
            val_rows.append(rows[i])
        elif i in test_idxs:
            test_rows.append(rows[i])
        else:
            train_rows.append(rows[i])
    return train_rows, val_rows, test_rows


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if training_args.ddp_find_unused_parameters:
        kwargs = [DistributedDataParallelKwargs(dim=0, broadcast_buffers=True, bucket_cap_mb=25, find_unused_parameters=True, check_reduction=False, gradient_as_bucket_view=False)]
    else:
        kwargs = []
    accelerator = Accelerator(kwargs_handlers=kwargs)

    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    experiment_id = custom_args.experiment_id or generate_experiment_id(
        name="DermQA",
        split="train",
        model_name=(model_args.model_name_or_path if "/" not in model_args.model_name_or_path else model_args.model_name_or_path.split("/")[-1]),
        pooling_mode=model_args.pooling_mode,
        train_batch_size=training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps,
        max_seq_length=model_args.max_seq_length,
        bidirectional=model_args.bidirectional,
        epochs=training_args.num_train_epochs,
        seed=training_args.seed,
        warmup_steps=training_args.warmup_steps,
        lr=training_args.learning_rate,
        lora_r=custom_args.lora_r,
    )

    training_args.output_dir = f"{training_args.output_dir}/{experiment_id}"

    if data_args.train_file and data_args.validation_file and data_args.test_file:
        train_rows = QACsvDataset(file_path=data_args.train_file, separator=data_args.separator).rows
        val_rows = QACsvDataset(file_path=data_args.validation_file, separator=data_args.separator).rows
        test_rows = QACsvDataset(file_path=data_args.test_file, separator=data_args.separator).rows
    else:
        if data_args.data_file is None:
            raise ValueError("Must provide either train/validation/test files or a single data_file for splitting.")
        all_rows = QACsvDataset(file_path=data_args.data_file, separator=data_args.separator).rows
        train_rows, val_rows, test_rows = split_rows(all_rows, data_args.val_ratio, data_args.test_ratio, training_args.seed)

    train_dataset = QACsvDataset(rows=train_rows, split="train", separator=data_args.separator)
    eval_dataset = QACsvDataset(rows=val_rows, split="validation", separator=data_args.separator) if len(val_rows) > 0 else None
    test_dataset = QACsvDataset(rows=test_rows, split="test", separator=data_args.separator) if len(test_rows) > 0 else None

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    model.model = initialize_peft(model.model, lora_r=custom_args.lora_r, lora_alpha=2 * custom_args.lora_r, lora_dropout=custom_args.lora_dropout)

    tokenizer = model.tokenizer

    from llm2vec.loss.utils import load_loss
    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    data_collator = DefaultCollator(model)

    metrics_fn = build_compute_metrics(ks=custom_args.recall_ks)

    trainer = LLM2VecSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=train_loss,
        compute_metrics=metrics_fn,
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    if training_args.do_train:
        trainer.train()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if getattr(training_args, "do_predict", False) and test_dataset is not None:
        results = trainer.predict(test_dataset, metric_key_prefix="test")
        metrics = results.metrics
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
