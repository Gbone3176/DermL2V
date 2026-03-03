#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import datasets
from datasets import load_dataset, load_metric

import torch
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, PredictionOutput
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from peft import PeftModel, LoraConfig, get_peft_model, TaskType

from llm2vec.llm2vec import LLM2Vec
from .trainer_seqcls import SeqClsTrainer


# Ensure minimal version
check_min_version("4.9.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    metric_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the metric"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch.",
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set.",
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set.",
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set.",
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models).",
        },
    )
    peft_addr: Optional[str] = field(
        default=None,
        metadata={"help": "Optional LoRA adapter path to load onto the base model."},
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": "Attention implementation to use in base model.",
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )


@dataclass
class CustomArguments:
    pooling_mode: str = field(
        default="mean",
        metadata={
            "help": "Pooling mode for sentence representation.",
            "choices": [
                "mean",
                "weighted_mean",
                "last_token",
                "eos_token",
                "bos_token",
                "latent_pooling",
                "eos",
                "weight",
            ],
        },
    )
    classifier_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability on classifier head."},
    )
    lora_r: Optional[int] = field(
        default=0,
        metadata={"help": "LoRA rank. Set >0 to enable LoRA fine-tuning."},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha (defaults to lora_r when not set)."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout applied to adapter (0.0–0.3 typical)."},
    )


class ModelForSeqCls(transformers.PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config: AutoConfig,
        encoder: LLM2Vec,
        classifier_dropout: float = 0.1,
    ):
        super().__init__(config)
        self.encoder = encoder
        self.dropout = nn.Dropout(classifier_dropout)
        in_dim = config.hidden_size
        self.classifier = nn.Linear(in_dim, config.num_labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, transformers.modeling_outputs.SequenceClassifierOutput]:
        f1 = {"input_ids": input_ids if input_ids is not None else kwargs.get("input_ids"),
              "attention_mask": attention_mask if attention_mask is not None else kwargs.get("attention_mask")}

        pooled = self.encoder(f1)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            if logits.size(-1) == 1 and labels.dim() == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif labels.dim() == 2:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=None, attentions=None
        )


def initialize_peft(model: transformers.PreTrainedModel, config: AutoConfig, lora_r: int, lora_alpha: Optional[int] = None, lora_dropout: float = 0.0):
    lora_alpha = lora_alpha if lora_alpha is not None else lora_r
    cls_name = config.__class__.__name__
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_cfg)


class SeqClsLLM2VecTrainer(SeqClsTrainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if output_dir is None or output_dir == "":
            raise ValueError("output_dir is empty; please set --output_dir")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Save classifier head state_dict and tokenizer (efficient)
        torch.save(self.model.classifier.state_dict(), os.path.join(output_dir, "classifier_state.pt"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Save training args for reproducibility
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save PEFT adapter weights if present
        try:
            inner = getattr(self.model, "model", None)
            if inner is not None and isinstance(inner, PeftModel):
                inner.save_pretrained(output_dir)
                logger.info("Saved PEFT adapter to output directory")
        except Exception as e:
            logger.warning(f"Failed to save PEFT adapter: {e}")


def _load_for_inference(model: ModelForSeqCls, output_dir: str):
    """Load classifier head and PEFT adapter from output_dir when evaluating/predicting without training."""
    if not output_dir or not os.path.isdir(output_dir):
        return
    # Load PEFT adapter if present
    adapter_cfg = os.path.join(output_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        try:
            inner = getattr(model, "model", None)
            if inner is not None and not isinstance(inner, PeftModel):
                inner = PeftModel.from_pretrained(inner, output_dir)
                model.model = inner
                logger.info("Loaded PEFT adapter from output_dir for inference")
        except Exception as e:
            logger.warning(f"Failed to load PEFT adapter from {output_dir}: {e}")
    # Load classifier head state_dict if present
    clf_state = os.path.join(output_dir, "classifier_state.pt")
    if os.path.exists(clf_state):
        try:
            state = torch.load(clf_state, map_location="cpu")
            model.classifier.load_state_dict(state)
            logger.info("Loaded classifier head state_dict from output_dir")
        except Exception as e:
            logger.warning(f"Failed to load classifier_state.pt: {e}")
    else:
        # Backward compatibility: try to load full classifier module
        clf_mod = os.path.join(output_dir, "classifier.pt")
        if os.path.exists(clf_mod):
            try:
                clf = torch.load(clf_mod, map_location="cpu")
                if isinstance(clf, nn.Module):
                    model.classifier.load_state_dict(clf.state_dict())
                    logger.info("Loaded classifier module and transferred state_dict")
            except Exception as e:
                logger.warning(f"Failed to load legacy classifier.pt: {e}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    # Normalize pooling aliases
    if custom_args.pooling_mode == "eos":
        custom_args.pooling_mode = "eos_token"
    if custom_args.pooling_mode == "weight":
        custom_args.pooling_mode = "weighted_mean"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Seed
    set_seed(training_args.seed)

    # Load datasets
    if data_args.task_name is not None:
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert test_extension == train_extension, "`test_file` should have same extension as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")
        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")
        if data_args.train_file.endswith(".csv"):
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        is_multiclass_binary = raw_datasets["train"].features["label"].dtype in ["list"]
        if is_regression:
            num_labels = 1
        elif is_multiclass_binary:
            # assert data_args.metric_name is not None and data_args.metric_name.startswith("hoc")
            num_labels = len(raw_datasets["train"][0]["label"])
            label_list = list(range(num_labels))
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    # Config & tokenizer
    cfg_addr = model_args.config_name if model_args.config_name else model_args.model_name_or_path
    config = AutoConfig.from_pretrained(
        cfg_addr,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Build llm2vec and base model
    l2v = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        peft_model_name_or_path=model_args.peft_addr,
        merge_peft=True,
        pooling_mode=custom_args.pooling_mode,
        skip_instruction=False,
        attn_implementation=model_args.attn_implementation,
        max_length=data_args.max_seq_length,
    )
    # base_model = l2v.model

    # Use llm2vec tokenizer for consistent special tokens and padding side
    tokenizer = l2v.tokenizer

    # Optionally apply LoRA to the inner HF model used by LLM2Vec
    if custom_args.lora_r and custom_args.lora_r > 0:
        if not isinstance(l2v.model, PeftModel):
            l2v.model = initialize_peft(
                l2v.model,
                config,
                lora_r=custom_args.lora_r,
                lora_alpha=custom_args.lora_alpha if custom_args.lora_alpha is not None else custom_args.lora_r,
                lora_dropout=custom_args.lora_dropout,
            )

    model = ModelForSeqCls(
        config=config,
        encoder=l2v,
        classifier_dropout=custom_args.classifier_dropout,
    )

    # Freeze encoder params except classifier head and LoRA adapter weights
    for n, p in list(model.named_parameters()):
        if ("classifier" in n) or ("lora_" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Data Padding & preprocess
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    sentence1_key, sentence2_key = "sentence", None

    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}.",
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.",
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)



    def preprocess_function(examples):
        texts = examples["sentence"]
        enc = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True)
        result = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        if "label" in examples:
            result["label"] = examples["label"]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Metrics
    def compute_metrics(p: EvalPrediction, eval_dataset):
        # Get the metric function
        if data_args.task_name is not None:
            metric = load_metric("glue", data_args.task_name)
        else:
            metric = load_metric("accuracy")

        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if data_args.metric_name == "hoc":
            from .utils_hoc import eval_hoc
            labels = np.array(p.label_ids).astype(int) #[num_ex, num_class]
            preds = (np.array(preds) > 0).astype(int)  #[num_ex, num_class]
            ids = eval_dataset["id"]
            return eval_hoc(labels.tolist(), preds.tolist(), list(ids))

        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif data_args.metric_name == "pearsonr":
            from scipy.stats import pearsonr as scipy_pearsonr
            pearsonr = float(scipy_pearsonr(p.label_ids, preds)[0])
            return {"pearsonr": pearsonr}
        elif data_args.metric_name == "PRF1":
            TP = ((preds == p.label_ids) & (preds != 0)).astype(int).sum().item()
            P_total = (preds != 0).astype(int).sum().item()
            L_total = (p.label_ids != 0).astype(int).sum().item()
            P = TP / P_total if P_total else 0
            R = TP / L_total if L_total else 0
            F1 = 2 * P * R / (P + R) if (P + R) else 0
            return {"precision": P, "recall": R, "F1": F1}
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

            
    def collate_fn(features):
        s = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        b = tokenizer.pad(s, padding="longest", return_tensors="pt")
        result = {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
        if "label" in features[0]:
            labels = torch.tensor([f["label"] for f in features], dtype=torch.float32)
            result["labels"] = labels
        return result

    # Trainer
    trainer = SeqClsLLM2VecTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        # If evaluating without training, attempt to load artifacts
        if not training_args.do_train:
            _load_for_inference(model, training_args.output_dir)
        logger.info("*** Evaluate ***")

        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])

        for ed, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=ed)
            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(ed)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(ed))
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if os.environ.get("USE_CODALAB", 0):
            import json
            json.dump(metrics, open("dev_stats.json", "w"))

    # Predict
    if training_args.do_predict:
        # If predicting without training, attempt to load artifacts
        if not training_args.do_train:
            _load_for_inference(model, training_args.output_dir)
        logger.info("*** Predict ***")
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for pd, task in zip(predict_datasets, tasks):
            results = trainer.predict(pd, metric_key_prefix="test")
            predictions = results.predictions
            metrics = results.metrics
            metrics["test_samples"] = len(pd)
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
            trainer.log(metrics)

            import json
            output_dir = training_args.output_dir
            output_path = f"{output_dir}/test_outputs_{task}.json" if task is not None else f"{output_dir}/test_outputs.json"
            json.dump({"predictions": results.predictions.tolist(), "label_ids": results.label_ids.tolist()}, open(output_path, "w"))

        if os.environ.get("USE_CODALAB", 0):
            import json
            json.dump(metrics, open("test_stats.json", "w"))

    # Push to hub
    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
        if data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"
        trainer.push_to_hub(**kwargs)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
