import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import datasets
import evaluate
from datasets import load_dataset

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils.versions import require_version

from llm2vec import LLM2Vec
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)
# 确保根日志有处理器，INFO 可见
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -U datasets",
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Base model or path for llm2vec."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Tokenizer name/path if different from model."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for HF downloads."})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Torch dtype override.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": "Attention implementation.",
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    classifier_dropout: Optional[float] = field(default=0.1, metadata={"help": "Dropout on classifier head."})
    peft_addr: Optional[str] = field(default=None, metadata={"help": "LoRA adapter path to load (optional)."})
    merge_subwords: bool = field(default=True, metadata={"help": "Average subword representations to first subword."})
    bidirectional: bool = field(default=True, metadata={"help": "Enable bidirectional attention in llm2vec."})


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Hugging Face dataset name (e.g., conll2003)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "Dataset config name (e.g., en)."})
    train_file: Optional[str] = field(default=None, metadata={"help": "Path to training data file (JSON/JSONL)."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Path to validation data file (JSON/JSONL)."})
    test_file: Optional[str] = field(default=None, metadata={"help": "Path to test data file (JSON/JSONL)."})
    overwrite_cache: bool = field(default=True, metadata={"help": "Whether to overwrite HF cache."})
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "Max sequence length after tokenization."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Num workers for preprocessing."})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Pad to max length vs dynamic pad."})
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode for datasets."})


@dataclass
class CustomArguments:
    lora_r: Optional[int] = field(default=0, metadata={"help": "LoRA rank; set >0 to enable."})
    lora_alpha: Optional[int] = field(default=None, metadata={"help": "LoRA alpha; defaults to lora_r if None."})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout."})
    retroactive_labels: str = field(
        default="same_token",
        metadata={"help": "Label alignment: same_token or next_token.", "choices": ["same_token", "next_token"]},
    )


class ModelForNER(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, base_model, merge_subwords: bool = True, classifier_dropout: float = 0.1):
        super().__init__(config)
        self.model = base_model
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.merge_subwords = merge_subwords

    def _merge_subwords(self, hidden_states, token_type_ids, attention_mask):
        # token_type_ids is word index per token; average subword vectors to the first subword position
        batch_size, seq_len, hidden = hidden_states.shape
        merged = hidden_states.clone()
        for b in range(batch_size):
            # collect indices by word id
            word_to_positions: Dict[int, List[int]] = {}
            for t in range(seq_len):
                if attention_mask[b, t].item() == 0:
                    continue
                wid = int(token_type_ids[b, t].item()) if token_type_ids is not None else -1
                if wid is None or wid < 0:
                    continue
                word_to_positions.setdefault(wid, []).append(t)
            # average per word and assign to first subword position
            for wid, positions in word_to_positions.items():
                if not positions:
                    continue
                vec = hidden_states[b, positions, :].mean(dim=0)
                first_pos = positions[0]
                merged[b, first_pos, :] = vec
        return merged

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, TokenClassifierOutput]:
        base_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k not in {"labels", "token_type_ids"}},
        )

        hidden_states = base_outputs[0] if isinstance(base_outputs, tuple) else base_outputs.last_hidden_state
        if self.merge_subwords and token_type_ids is not None:
            hidden_states = self._merge_subwords(hidden_states, token_type_ids, attention_mask)
        logits = self.classifier(self.dropout(hidden_states))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only compute loss on positions where labels != -100
            active_positions = labels.view(-1) != -100
            active_logits = logits.view(-1, self.config.num_labels)[active_positions]
            active_labels = labels.view(-1)[active_positions]
            loss = loss_fct(active_logits, active_labels)

        if isinstance(base_outputs, tuple):
            return (loss, logits) + base_outputs[2:]
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=base_outputs.hidden_states, attentions=base_outputs.attentions)


def initialize_peft(model, lora_r: int, lora_alpha: Optional[int], lora_dropout: float):
    alpha = lora_alpha if lora_alpha is not None else lora_r
    # Common target modules for decoder-based architectures
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    config = LoraConfig(
        r=lora_r,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="TOKEN_CLS",
    )
    return get_peft_model(model, config)


def build_label_mappings(raw_datasets, task_key: str):
    # For HF datasets like conll2003, use ClassLabel.names; for JSON, infer from split
    features = raw_datasets["train"].features
    if task_key in features and hasattr(features[task_key], "names"):
        names = features[task_key].names
    else:
        # Infer from validation/train labels if no ClassLabel
        labels = set()
        for split in ["train", "validation"]:
            if split in raw_datasets:
                for eg in raw_datasets[split]:
                    for lid in eg[task_key]:
                        labels.add(lid)
        # If labels are strings, sort; if ints, build placeholders
        if all(isinstance(l, str) for l in labels):
            names = sorted(labels)
        else:
            # Fallback to unique ints mapped to strings
            names = [str(i) for i in sorted(labels)]
    id2label = {i: name for i, name in enumerate(names)}
    label2id = {name: i for i, name in enumerate(names)}
    return names, id2label, label2id


def tokenize_and_align_labels(
    examples: Dict[str, Any],
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    task_key: str,
    max_seq_length: Optional[int],
    pad_to_max_length: bool,
    retroactive_labels: str,
):
    is_next = retroactive_labels == "next_token"
    padding = "max_length" if pad_to_max_length else False
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
        padding=padding,
    )

    labels = []
    token_type_ids = []  # word indices per token; -1 for special/padding
    for i, tokens in enumerate(examples["tokens"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        example_labels = examples[task_key][i]
        # map labels from names to ids if necessary
        if len(example_labels) and isinstance(example_labels[0], str):
            example_labels = [label2id[x] for x in example_labels]

        aligned_labels = []
        aligned_word_indices = []
        prev_wid = None
        for pos, wid in enumerate(word_ids):
            if wid is None:
                aligned_labels.append(-100)
                aligned_word_indices.append(-1)
                prev_wid = wid
                continue
            # next_token: shift labels by one position (predict next token label at current)
            effective_wid = wid + 1 if is_next else wid
            # first subword gets label; subsequent subwords -> -100
            if wid != prev_wid:
                if 0 <= effective_wid < len(example_labels):
                    aligned_labels.append(example_labels[effective_wid])
                else:
                    aligned_labels.append(-100)
                aligned_word_indices.append(wid)
            else:
                aligned_labels.append(-100)
                aligned_word_indices.append(wid)
            prev_wid = wid

        labels.append(aligned_labels)
        token_type_ids.append(aligned_word_indices)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["token_type_ids"] = token_type_ids
    return tokenized_inputs


def compute_metrics(config_kwargs: Dict[str, Any]):
    seqeval = evaluate.load("seqeval")

    def _compute(p):
        predictions = p.predictions
        labels = p.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        # argmax over class dimension
        predictions = np.argmax(predictions, axis=2)
        id2label = config_kwargs["id2label"]

        true_predictions = []
        true_labels = []
        for pred_seq, lab_seq in zip(predictions, labels):
            preds = []
            labs = []
            for p_i, l_i in zip(pred_seq, lab_seq):
                if l_i == -100:
                    continue
                preds.append(id2label[int(p_i)])
                labs.append(id2label[int(l_i)])
            true_predictions.append(preds)
            true_labels.append(labs)

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        # Overall metrics for compatibility
        overall_precision = results.get("overall_precision", results.get("precision", 0.0))
        overall_recall = results.get("overall_recall", results.get("recall", 0.0))
        overall_f1 = results.get("overall_f1", results.get("f1", 0.0))
        overall_accuracy = results.get("overall_accuracy", results.get("accuracy", 0.0))

        # Macro metrics (mean over entity types, skip 'overall*')
        Ps, Rs, Fs = [], [], []
        for key, val in results.items():
            if key.startswith("overall"):
                continue
            if isinstance(val, dict):
                if "precision" in val and "recall" in val and "f1" in val:
                    Ps.append(val["precision"])
                    Rs.append(val["recall"])
                    Fs.append(val["f1"])
        macro_precision = float(np.mean(Ps)) if Ps else 0.0
        macro_recall = float(np.mean(Rs)) if Rs else 0.0
        macro_f1 = float(np.mean(Fs)) if Fs else 0.0

        return {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "accuracy": overall_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    return _compute


def try_load_dataset(data_args: DataTrainingArguments, token: Optional[str]):
    # Prefer Hub datasets; fall back to JSON/JSONL files
    if data_args.dataset_name:
        name = data_args.dataset_name
        config = data_args.dataset_config_name
        # Normalize local script reference like "conll2003.py"
        if name.endswith(".py"):
            base = os.path.basename(name)
            name = base[:-3]
        # Default config for conll2003
        if name == "conll2003" and not config:
            config = "en"
        try:
            return load_dataset(name, config, cache_dir=None, token=token, streaming=data_args.streaming)
        except RuntimeError as e:
            msg = str(e)
            if "Dataset scripts are no longer supported" in msg:
                raise RuntimeError(
                    "datasets>=3 禁止从本地 .py 脚本加载数据集。请改用 `dataset_name='conll2003'` 或提供 JSON/JSONL 文件路径。"
                )
            raise
    else:
        data_files = {}
        if data_args.train_file:
            data_files["train"] = data_args.train_file
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file:
            data_files["test"] = data_args.test_file
        if not data_files:
            raise ValueError("必须提供 `dataset_name` 或 `train_file/validation_file`。")
        # Only support JSON/JSONL structured as {tokens: [...], ner_tags: [...]} per example
        ext = os.path.splitext(list(data_files.values())[0])[1].lower()
        if ext in [".json", ".jsonl"]:
            return load_dataset("json", data_files=data_files, token=token, streaming=data_args.streaming)
        raise ValueError("仅支持 JSON/JSONL 文件格式（包含 'tokens' 与 'ner_tags' 字段）。")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments))
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
    send_telemetry = getattr(training_args, "report_to", None)
    set_seed(training_args.seed)

    # Load dataset
    raw_datasets = try_load_dataset(data_args, token=getattr(model_args, "token", None))
    task_key = "ner_tags"

    # Build tokenizer
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build labels
    names, id2label, label2id = build_label_mappings(raw_datasets, task_key)
    config_kwargs = {
        "num_labels": len(names),
        "id2label": id2label,
        "label2id": label2id,
    }

    # Config and base llm2vec model
    torch_dtype = (
        torch.float32
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    l2v = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_addr,
        merge_peft=True,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    # Build config for classifier
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # Optionally inject LoRA
    base_model = l2v.model
    if custom_args.lora_r and custom_args.lora_r > 0:
        if not isinstance(base_model, PeftModel):
            alpha = custom_args.lora_alpha if custom_args.lora_alpha is not None else custom_args.lora_r
            base_model = initialize_peft(base_model, lora_r=custom_args.lora_r, lora_alpha=alpha, lora_dropout=custom_args.lora_dropout)

    model = ModelForNER(
        config=config,
        base_model=base_model,
        merge_subwords=model_args.merge_subwords,
        classifier_dropout=model_args.classifier_dropout,
    )

    # Freeze base params except classifier and LoRA
    for n, p in list(model.named_parameters()):
        if ("classifier" in n) or ("lora_" in n):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Max seq length
    if data_args.max_seq_length is None:
        max_seq_length = min(tokenizer.model_max_length, 1024)
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"max_seq_length({data_args.max_seq_length}) 超过 tokenizer.model_max_length({tokenizer.model_max_length})，使用后者。"
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocess function
    def _tokenize_and_align(examples):
        return tokenize_and_align_labels(
            examples,
            tokenizer,
            label2id,
            task_key,
            max_seq_length,
            data_args.pad_to_max_length,
            custom_args.retroactive_labels,
        )

    with training_args.main_process_first(desc="dataset map pre-processing"):
        # Determine base columns from available split
        base_split = "train" if "train" in raw_datasets else list(raw_datasets.keys())[0]
        base_columns = raw_datasets[base_split].column_names
        processed_datasets = raw_datasets.map(
            _tokenize_and_align,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=base_columns,
            desc="Tokenizing and aligning labels",
        )
    # 打印模型的可训练参数以及全部参数
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # 打印可训练模块名称
        logger.info("可训练模块：")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}")
        logger.info(f"全部参数: {total:,}, 可训练参数: {trainable:,}, 可训练参数占比: {trainable / total:.5%}")

    count_parameters(model)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Metrics
    metrics_fn = compute_metrics(config_kwargs)



    class NERTrainer(Trainer):
        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")
        
            # 保存分类头、tokenizer、训练参数
            torch.save(self.model.classifier, os.path.join(output_dir, "classifier.pt"))
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
            # 如果内层模型是 PEFT，则保存 LoRA 适配器
            try:
                inner = getattr(self.model, "model", None)
                if inner is not None and isinstance(inner, PeftModel):
                    inner.save_pretrained(output_dir)
                    logger.info("Saved PEFT adapter to output directory")
            except Exception as e:
                logger.warning(f"Failed to save PEFT adapter: {e}")
        
            # 如果内层模型是 PEFT，则保存 LoRA 适配器
        
            # If inner model is PEFT, save LoRA adapter
        
        
    # Trainer
    trainer = NERTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets.get("train"),
        eval_dataset=processed_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
    )

    # Train
    if training_args.do_train:
        trainer.train()
        # 使用自定义 _save 保存分类头、tokenizer，以及（若存在）PEFT 适配器
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluate
    if training_args.do_eval and "validation" in processed_datasets:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if getattr(training_args, "do_predict", False) and "test" in processed_datasets:
        predict_dataset = processed_datasets["test"]
        results = trainer.predict(predict_dataset, metric_key_prefix="test")
        predictions = results.predictions
        metrics = results.metrics
        metrics["test_samples"] = len(predict_dataset)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save raw outputs for analysis
        output_dir = training_args.output_dir
        try:
            word_ids = predict_dataset["token_type_ids"]
        except Exception:
            word_ids = None
        import json
        with open(os.path.join(output_dir, "test_outputs.json"), "w") as f:
            json.dump(
                {
                    "predictions": predictions.tolist() if hasattr(predictions, "tolist") else predictions,
                    "label_ids": results.label_ids.tolist() if hasattr(results.label_ids, "tolist") else results.label_ids,
                    "word_ids": word_ids,
                },
                f,
            )

    elif getattr(training_args, "do_predict", False):
        logger.warning("指定了 do_predict，但未提供测试集（无 'test' split）。")


if __name__ == "__main__":
    main()