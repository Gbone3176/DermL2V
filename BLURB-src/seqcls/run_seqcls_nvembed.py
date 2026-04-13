#!/usr/bin/env python
# coding=utf-8

import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, HfArgumentParser, LlamaTokenizerFast


logger = logging.getLogger(__name__)

NVEMBED_LOCAL_PATH = "/cache/modelscope/models/nv-community/NV-Embed-v2"
NVEMBED_MODEL_IDS = {
    "nvidia/NV-Embed-v2",
    NVEMBED_LOCAL_PATH,
}


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "NV-Embed-v2 model path or HF id."})
    model_name: Optional[str] = field(default=None, metadata={"help": "Name used for output folder/table."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Optional cache dir."})
    attn_implementation: str = field(default="eager", metadata={"help": "Attention implementation."})
    instruction: Optional[str] = field(default=None, metadata={"help": "Optional encoding instruction."})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Training json/jsonl file."})
    validation_file: str = field(metadata={"help": "Validation json/jsonl file."})
    test_file: str = field(metadata={"help": "Test json/jsonl file."})
    metric_name: str = field(default="All", metadata={"help": "Metric bundle name."})
    max_seq_length: int = field(default=512, metadata={"help": "Encoding max length."})


@dataclass
class TrainingArguments:
    output_dir: str = field(metadata={"help": "Output directory root."})
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite output dir."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    do_train: bool = field(default=True, metadata={"help": "Run train."})
    do_eval: bool = field(default=True, metadata={"help": "Run dev eval."})
    do_predict: bool = field(default=True, metadata={"help": "Run test eval."})
    encode_batch_size: int = field(default=4, metadata={"help": "Batch size for frozen encoder."})
    per_device_train_batch_size: int = field(default=64, metadata={"help": "Batch size for linear head."})
    learning_rate: float = field(default=1e-3, metadata={"help": "Head learning rate."})
    weight_decay: float = field(default=1e-2, metadata={"help": "Head weight decay."})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio."})
    num_train_epochs: int = field(default=40, metadata={"help": "Head training epochs."})
    logging_steps: int = field(default=10, metadata={"help": "Unused compatibility arg."})
    eval_steps: int = field(default=1, metadata={"help": "Unused compatibility arg."})
    fp16: bool = field(default=True, metadata={"help": "Use fp16 for encoder if possible."})
    bf16: bool = field(default=False, metadata={"help": "Use bf16 for encoder if possible."})
    metric_for_best_model: str = field(default="f1_macro", metadata={"help": "Dev metric used for selection."})
    linear_probing: bool = field(default=True, metadata={"help": "Compatibility flag; always true here."})
    report_to: str = field(default="none", metadata={"help": "Compatibility arg."})


def patch_nvembed_tokenizer_loading():
    original_from_pretrained = AutoTokenizer.from_pretrained

    def wrapped_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        target = str(pretrained_model_name_or_path)
        if target in NVEMBED_MODEL_IDS:
            local_snapshot = target if os.path.isdir(target) else NVEMBED_LOCAL_PATH
            return LlamaTokenizerFast.from_pretrained(local_snapshot)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoTokenizer.from_pretrained = wrapped_from_pretrained


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = sigmoid(logits)
    preds = (logits > 0).astype(int)
    metrics = {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
    }
    try:
        metrics["mAP_macro"] = average_precision_score(labels, probs, average="macro")
        metrics["mAP_micro"] = average_precision_score(labels, probs, average="micro")
    except ValueError:
        pass
    try:
        metrics["roc_auc_macro"] = roc_auc_score(labels, probs, average="macro")
        metrics["roc_auc_micro"] = roc_auc_score(labels, probs, average="micro")
    except ValueError:
        pass
    return metrics


def build_instruction_prefix(instruction: Optional[str]) -> str:
    if not instruction:
        return ""
    return f"Instruct: {instruction}\nQuery: "


def add_eos(texts: List[str], eos_token: str) -> List[str]:
    return [text + eos_token for text in texts]


@torch.no_grad()
def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    instruction: Optional[str],
) -> np.ndarray:
    backend = model._first_module().auto_model
    prompt = build_instruction_prefix(instruction)
    eos_token = getattr(model.tokenizer, "eos_token", None) or ""
    if eos_token:
        texts = add_eos(texts, eos_token)

    if hasattr(backend, "_do_encode"):
        embeddings = backend._do_encode(
            texts,
            batch_size=batch_size,
            instruction=prompt,
            max_length=max_length,
            num_workers=0,
            return_numpy=True,
        )
    elif hasattr(backend, "encode"):
        embeddings = backend.encode(
            texts,
            instruction=prompt,
            max_length=max_length,
        )
    else:
        kwargs = {
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": True,
            "normalize_embeddings": True,
        }
        if prompt:
            kwargs["prompt"] = prompt
        embeddings = model.encode(texts, **kwargs)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype=np.float32)


def evaluate_head(head: nn.Module, features: np.ndarray, labels: np.ndarray, device: torch.device) -> Dict[str, float]:
    head.eval()
    with torch.no_grad():
        feats = torch.from_numpy(features).to(device)
        logits = head(feats).cpu().numpy()
    return compute_metrics(labels, logits)


def predict_head(head: nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        feats = torch.from_numpy(features).to(device)
        return head(feats).cpu().numpy()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    if not train_args.linear_probing:
        raise ValueError("This script only supports linear probing with a frozen encoder.")

    model_name = model_args.model_name or model_args.model_name_or_path.rstrip("/").split("/")[-1]
    output_dir = train_args.output_dir
    if not output_dir.endswith(model_name):
        output_dir = os.path.join(output_dir, model_name)
    train_args.output_dir = output_dir

    if os.path.isdir(output_dir) and os.listdir(output_dir) and not train_args.overwrite_output_dir:
        raise ValueError(f"Output directory exists and is not empty: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    set_seed(train_args.seed)
    patch_nvembed_tokenizer_loading()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if torch.cuda.is_available():
        if train_args.bf16:
            dtype = torch.bfloat16
        elif train_args.fp16:
            dtype = torch.float16

    model_kwargs = {
        "attn_implementation": model_args.attn_implementation,
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    logger.info("Loading NV-Embed model from %s", model_args.model_name_or_path)
    encoder = SentenceTransformer(
        model_args.model_name_or_path,
        device=str(device),
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left", "trust_remote_code": True},
    )
    encoder.max_seq_length = data_args.max_seq_length

    train_rows = load_jsonl(data_args.train_file)
    dev_rows = load_jsonl(data_args.validation_file)
    test_rows = load_jsonl(data_args.test_file)

    train_texts = [row["sentence"] for row in train_rows]
    dev_texts = [row["sentence"] for row in dev_rows]
    test_texts = [row["sentence"] for row in test_rows]
    y_train = np.asarray([row["label"] for row in train_rows], dtype=np.float32)
    y_dev = np.asarray([row["label"] for row in dev_rows], dtype=np.float32)
    y_test = np.asarray([row["label"] for row in test_rows], dtype=np.float32)

    logger.info("Encoding train/dev/test splits with frozen encoder")
    x_train = encode_texts(encoder, train_texts, train_args.encode_batch_size, data_args.max_seq_length, model_args.instruction)
    x_dev = encode_texts(encoder, dev_texts, train_args.encode_batch_size, data_args.max_seq_length, model_args.instruction)
    x_test = encode_texts(encoder, test_texts, train_args.encode_batch_size, data_args.max_seq_length, model_args.instruction)

    np.save(os.path.join(output_dir, "train_embeddings.npy"), x_train)
    np.save(os.path.join(output_dir, "dev_embeddings.npy"), x_dev)
    np.save(os.path.join(output_dir, "test_embeddings.npy"), x_test)

    in_dim = x_train.shape[1]
    num_labels = y_train.shape[1]
    head = nn.Linear(in_dim, num_labels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(x_train) / train_args.per_device_train_batch_size)
    total_steps = max(1, steps_per_epoch * train_args.num_train_epochs)
    warmup_steps = int(total_steps * train_args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(train_args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_args.per_device_train_batch_size,
        shuffle=True,
        generator=generator,
    )

    log_history = []
    best_metric = float("-inf")
    best_epoch = 0.0
    best_state = None
    global_step = 0

    for epoch in range(train_args.num_train_epochs):
        head.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_losses.append(loss.item())

        epoch_float = float(epoch + 1)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_log = {
            "epoch": epoch_float,
            "loss": train_loss,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "step": global_step,
        }
        log_history.append(train_log)
        logger.info("epoch=%s train_loss=%.6f", epoch + 1, train_loss)

        if train_args.do_eval:
            eval_metrics = evaluate_head(head, x_dev, y_dev, device)
            eval_log = {"epoch": epoch_float, "step": global_step}
            for key, value in eval_metrics.items():
                eval_log[f"eval_{key}"] = float(value)
            log_history.append(eval_log)
            with open(os.path.join(output_dir, "eval_history.jsonl"), "a") as f:
                f.write(json.dumps(eval_log) + "\n")

            metric_value = eval_metrics.get(train_args.metric_for_best_model)
            if metric_value is None:
                raise ValueError(f"metric_for_best_model={train_args.metric_for_best_model} missing in eval metrics")
            if metric_value > best_metric:
                best_metric = float(metric_value)
                best_epoch = epoch_float
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)

    torch.save(head.state_dict(), os.path.join(output_dir, "classifier_state.pt"))

    train_results = {
        "epoch": best_epoch if train_args.do_eval else float(train_args.num_train_epochs),
        "train_loss": float(log_history[-2]["loss"] if train_args.do_eval and len(log_history) >= 2 else log_history[-1]["loss"]),
        "train_samples": len(train_rows),
    }
    with open(os.path.join(output_dir, "train_results.json"), "w") as f:
        json.dump(train_results, f, indent=4)

    if train_args.do_eval:
        dev_logits = predict_head(head, x_dev, device)
        dev_metrics = compute_metrics(y_dev, dev_logits)
        eval_results = {"epoch": best_epoch, "eval_samples": len(dev_rows)}
        for key, value in dev_metrics.items():
            eval_results[f"eval_{key}"] = float(value)
        with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=4)

    if train_args.do_predict:
        test_logits = predict_head(head, x_test, device)
        test_metrics = compute_metrics(y_test, test_logits)
        test_results = {"epoch": best_epoch, "test_samples": len(test_rows)}
        for key, value in test_metrics.items():
            test_results[f"test_{key}"] = float(value)
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)

    trainer_state = {
        "best_metric": best_metric if best_metric != float("-inf") else None,
        "best_model_checkpoint": output_dir,
        "epoch": best_epoch if train_args.do_eval else float(train_args.num_train_epochs),
        "eval_steps": train_args.eval_steps,
        "global_step": global_step,
        "is_hyper_param_search": False,
        "is_local_process_zero": True,
        "is_world_process_zero": True,
        "log_history": log_history,
        "max_steps": total_steps,
        "num_train_epochs": train_args.num_train_epochs,
        "save_steps": 0,
        "train_batch_size": train_args.per_device_train_batch_size,
        "trial_name": None,
        "trial_params": None,
    }
    with open(os.path.join(output_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f, indent=2)

    run_config = {
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "training_args": vars(train_args),
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)


if __name__ == "__main__":
    main()
