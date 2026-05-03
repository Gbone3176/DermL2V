#!/usr/bin/env python
"""LoRA fine-tuning for Qwen3-Embedding-8B on DermVariants triplets."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import swanlab
except ImportError:  # pragma: no cover - optional runtime dependency
    swanlab = None

THIS_FILE = Path(__file__).resolve()
CONTRASTIVE_ROOT = THIS_FILE.parents[1]
if str(CONTRASTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRASTIVE_ROOT))

from shared.dermvariants import DermTripletDataset, collate_triplets, load_dermvariants_triplets
from shared.losses import mismatched_sizes_all_gather


LOGGER = logging.getLogger("train_qwen3embedding8b_lora")
SWANLAB_ACTIVE = False

DEFAULT_MODEL_PATH = "/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B"
DEFAULT_DATA_DIR = "/storage/dataset/dermatoscop/Derm1M/DermVariantsData"
DEFAULT_OUTPUT_ROOT = "ContrastiveModel/Qwen3Embedding8B/output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dermqa_upsample_ratio", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint-* or final directory containing adapter and trainer_state.pt.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument("--swanlab_project", default="Contrastive Model fine-tune")
    parser.add_argument("--swanlab_run_name", default=None)
    parser.add_argument("--disable_swanlab", action="store_true")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--eval_include_negatives", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist_info = init_distributed()
    setup_logging(dist_info["is_main"])
    set_seed(args.seed + dist_info["rank"])

    if not args.fp16:
        LOGGER.warning("Qwen3-Embedding-8B LoRA defaults to fp16 on V100; continuing with --no-fp16.")

    device = torch.device("cuda", dist_info["local_rank"]) if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.info("Using device=%s rank=%s world_size=%s", device, dist_info["rank"], dist_info["world_size"])

    samples = load_dermvariants_triplets(
        data_dir=args.data_dir,
        split=args.split,
        effective_batch_size=args.per_device_batch_size * max(1, dist_info["world_size"]),
        seed=args.seed,
        dermqa_upsample_ratio=args.dermqa_upsample_ratio,
        use_query_instruction=False,
        max_samples=args.max_train_samples,
    )
    dataset = DermTripletDataset(samples)
    LOGGER.info("Loaded %s DermVariants triplets", len(dataset))

    sampler = (
        DistributedSampler(dataset, num_replicas=dist_info["world_size"], rank=dist_info["rank"], shuffle=True, seed=args.seed)
        if dist_info["distributed"]
        else RandomSampler(dataset)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        collate_fn=collate_triplets,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise ValueError("No full training batch is available. Lower --per_device_batch_size or increase data size.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        local_files_only=args.local_files_only,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint)
    run_dir = resume_checkpoint.parent if resume_checkpoint else build_run_dir(args)

    model = Qwen3EmbeddingModel(
        model_name_or_path=args.model_name_or_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=parse_lora_targets(args.lora_target_modules),
        fp16=args.fp16,
        normalize=not args.no_normalize,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        adapter_path=resume_checkpoint,
    ).to(device)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    log_trainable_parameters(model)

    if dist_info["distributed"]:
        model = DistributedDataParallel(model, device_ids=[dist_info["local_rank"]] if torch.cuda.is_available() else None)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(dataloader) // args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, math.ceil(args.num_train_epochs * steps_per_epoch))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = Qwen3HardNegativeLoss(temperature=args.temperature)

    resume_state = None
    if resume_checkpoint is not None:
        resume_state = load_training_state(resume_checkpoint, optimizer, scheduler, device, dist_info)

    eval_samples = None
    if dist_info["is_main"]:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "train_args.json", vars(args))
        write_json(
            run_dir / "run_info.json",
            {
                "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "num_samples": len(dataset),
                "world_size": dist_info["world_size"],
                "global_micro_batch_size": args.per_device_batch_size * dist_info["world_size"],
                "effective_batch_size": args.per_device_batch_size
                * args.gradient_accumulation_steps
                * dist_info["world_size"],
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "pooling": "last_token",
                "normalize": not args.no_normalize,
                "text_format": "bmretriever_query_passage_instruction",
                "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            },
        )
        eval_samples = load_dermvariants_triplets(
            data_dir=args.data_dir,
            split=args.eval_split,
            effective_batch_size=1,
            shuffle_individual_datasets=False,
            seed=args.seed,
            dermqa_upsample_ratio=1,
            use_query_instruction=False,
            max_samples=args.eval_max_samples,
        )
        LOGGER.info("Loaded %s eval triplets from split=%s", len(eval_samples), args.eval_split)
        init_swanlab(args, run_dir, dist_info)

    eval_steps = args.eval_every_steps if args.eval_every_steps > 0 else steps_per_epoch
    train(
        args=args,
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        sampler=sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        total_steps=total_steps,
        dist_info=dist_info,
        run_dir=run_dir,
        eval_samples=eval_samples,
        eval_steps=eval_steps,
        resume_state=resume_state,
    )

    if dist_info["is_main"]:
        final_state = {
            "global_step": total_steps,
            "epoch": resume_state.get("epoch", 0) if resume_state else 0,
            "next_batch_in_epoch": 0,
            "total_steps": total_steps,
            "completed": True,
        }
        save_checkpoint(model, tokenizer, run_dir / "final", args, optimizer, scheduler, final_state)
        LOGGER.info("Saved final LoRA checkpoint to %s", run_dir / "final")
        finish_swanlab()

    cleanup_distributed()


class Qwen3EmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list[str],
        fp16: bool,
        normalize: bool,
        attn_implementation: str,
        local_files_only: bool,
        adapter_path: Path | None = None,
    ):
        super().__init__()
        dtype = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
        base_model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        base_model.config.use_cache = False
        if adapter_path is not None:
            self.encoder = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.encoder = get_peft_model(base_model, peft_config)
        self.normalize = normalize

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask).float()
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


class Qwen3HardNegativeLoss(nn.Module):
    """InfoNCE over normalized Qwen3 embeddings with one row-aligned hard negative."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query: Tensor, positive: Tensor, negative: Tensor | None = None) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            query = torch.cat(mismatched_sizes_all_gather(query), dim=0)
            positive = torch.cat(mismatched_sizes_all_gather(positive), dim=0)
            if negative is not None:
                negative = torch.cat(mismatched_sizes_all_gather(negative), dim=0)

        if query.size(0) != positive.size(0):
            raise ValueError(f"query and positive must align row-wise, got {query.size(0)} and {positive.size(0)}")

        # Keep fp16 activations for memory, but compute logits in fp32.
        query = query.float()
        positive = positive.float()
        logits_pos = query @ positive.t()
        logits = logits_pos / self.temperature

        if negative is not None and negative.size(0) > 0:
            if negative.size(0) != query.size(0):
                raise ValueError(f"negative must align row-wise with query, got {negative.size(0)} and {query.size(0)}")
            negative = negative.float()
            logits_neg = ((query * negative).sum(dim=-1) / self.temperature).unsqueeze(1)
            logits = torch.cat([logits, logits_neg], dim=1)

        labels = torch.arange(query.size(0), dtype=torch.long, device=query.device)
        return self.cross_entropy(logits, labels)


def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer,
    dataloader: DataLoader,
    sampler,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: Qwen3HardNegativeLoss,
    device: torch.device,
    total_steps: int,
    dist_info: dict,
    run_dir: Path,
    eval_samples: list | None,
    eval_steps: int,
    resume_state: dict | None = None,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    resume_state = resume_state or {}
    global_step = int(resume_state.get("global_step", 0))
    running_loss = 0.0
    epoch = int(resume_state.get("epoch", 0))
    start_batch_in_epoch = int(resume_state.get("next_batch_in_epoch", 0))
    if dist_info["is_main"] and global_step > 0:
        LOGGER.info(
            "Resuming training at global_step=%s epoch=%s next_batch_in_epoch=%s",
            global_step,
            epoch,
            start_batch_in_epoch,
        )
    progress = tqdm(total=total_steps, initial=global_step, disable=not dist_info["is_main"], desc="training")

    while global_step < total_steps:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            if start_batch_in_epoch and step < start_batch_in_epoch:
                continue
            should_step = (step + 1) % args.gradient_accumulation_steps == 0
            sync_context = (
                model.no_sync()
                if dist_info["distributed"] and not should_step and hasattr(model, "no_sync")
                else nullcontext()
            )
            with sync_context:
                query, positive, negative = encode_triplets(model, tokenizer, batch, args.max_length, device)
                loss = loss_fn(query, positive, negative)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            running_loss += loss.detach().float().item()
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if dist_info["is_main"]:
                avg_loss = running_loss
                running_loss = 0.0
                progress.update(1)
                progress.set_postfix(loss=f"{avg_loss:.4f}")
                if global_step % args.logging_steps == 0 or global_step == 1:
                    LOGGER.info("step=%s loss=%.6f lr=%.6g", global_step, avg_loss, scheduler.get_last_lr()[0])
                    log_swanlab({"train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    train_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "next_batch_in_epoch": step + 1,
                        "total_steps": total_steps,
                        "completed": False,
                    }
                    save_checkpoint(
                        model,
                        tokenizer,
                        run_dir / f"checkpoint-{global_step}",
                        args,
                        optimizer,
                        scheduler,
                        train_state,
                    )
                if eval_samples and eval_steps > 0 and global_step % eval_steps == 0:
                    run_validation_eval(args, model, tokenizer, eval_samples, device, global_step, dist_info, run_dir)

            if dist_info["distributed"]:
                dist.barrier()
            if global_step >= total_steps:
                break
        epoch += 1
        start_batch_in_epoch = 0

    progress.close()


def encode_triplets(
    model,
    tokenizer,
    batch: dict[str, list[str]],
    max_length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = len(batch["query"])
    query_texts = [format_query(task_name, text) for task_name, text in zip(batch["task_name"], batch["query"])]
    positive_texts = [format_passage(text) for text in batch["positive"]]
    negative_texts = [format_passage(text) for text in batch["negative"]]
    embeddings = encode_texts(model, tokenizer, query_texts + positive_texts + negative_texts, max_length, device)
    query = embeddings[:batch_size]
    positive = embeddings[batch_size : 2 * batch_size]
    negative = embeddings[2 * batch_size :]
    return query, positive, negative


def encode_texts(model, tokenizer, texts: list[str], max_length: int, device: torch.device) -> Tensor:
    encoded = encode_with_eos(tokenizer, texts, max_length=max_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded)


def encode_with_eos(tokenizer, texts: list[str], max_length: int) -> dict[str, Tensor]:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def last_token_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


def format_query(task_name: str, text: str) -> str:
    instruction = {
        "SemVariants": "Retrieve the dermatology description that has the same clinical meaning as the query.",
        "VisVariants": "Retrieve the visual dermatology description that best matches the diagnosis-style query.",
        "DermQA": "Retrieve the answer that best responds to the dermatology question.",
        "SI1": "Retrieve the answer that best matches the dermatology clinical scenario.",
        "SI2": "Retrieve the medically correct answer passage for the dermatology question.",
    }.get(task_name, "Retrieve the most relevant dermatology passage for the query.")
    return f"{instruction}\nQuery: {text}"


def format_passage(text: str) -> str:
    return f"Represent this passage\npassage: {text}"


def run_validation_eval(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer,
    eval_samples: list,
    device: torch.device,
    global_step: int,
    dist_info: dict,
    run_dir: Path,
) -> None:
    if not dist_info["is_main"]:
        return
    eval_model = model.module if hasattr(model, "module") else model
    metrics = evaluate_retrieval(
        model=eval_model,
        tokenizer=tokenizer,
        samples=eval_samples,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
        include_negatives=args.eval_include_negatives,
    )
    eval_dir = run_dir / "eval" / f"step-{global_step}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_dir / "metrics.json", metrics)
    LOGGER.info("eval step=%s ndcg@10=%.6f recall@10=%.6f", global_step, metrics["ndcg@10"], metrics["recall@10"])
    log_swanlab({"eval/ndcg@10": metrics["ndcg@10"], "eval/recall@10": metrics["recall@10"]}, step=global_step)


def evaluate_retrieval(
    model: torch.nn.Module,
    tokenizer,
    samples: list,
    max_length: int,
    batch_size: int,
    device: torch.device,
    include_negatives: bool = True,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    query_texts = [format_query(sample.task_name, sample.query) for sample in samples]
    doc_texts = [format_passage(sample.positive) for sample in samples]
    relevant_doc_indices = list(range(len(doc_texts)))
    if include_negatives:
        doc_texts.extend(format_passage(sample.negative) for sample in samples)

    with torch.no_grad():
        query_embeddings = encode_texts_in_batches(model, tokenizer, query_texts, max_length, batch_size, device)
        doc_embeddings = encode_texts_in_batches(model, tokenizer, doc_texts, max_length, batch_size, device)

    scores = query_embeddings @ doc_embeddings.t()
    scores[torch.isnan(scores)] = -1
    top_k = min(10, scores.size(1))
    top_indices = torch.topk(scores, k=top_k, dim=1, largest=True, sorted=True).indices.cpu()

    recall_hits = 0
    ndcg_sum = 0.0
    for query_idx, doc_idx in enumerate(relevant_doc_indices):
        ranking = top_indices[query_idx].tolist()
        if doc_idx not in ranking:
            continue
        recall_hits += 1
        rank = ranking.index(doc_idx) + 1
        ndcg_sum += 1.0 / math.log2(rank + 1)

    num_queries = max(1, len(samples))
    metrics = {
        "ndcg@10": ndcg_sum / num_queries,
        "recall@10": recall_hits / num_queries,
        "num_queries": len(samples),
        "num_docs": len(doc_texts),
        "include_negatives": include_negatives,
    }
    if was_training:
        model.train()
    return metrics


def encode_texts_in_batches(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_embeddings = encode_texts(model, tokenizer, batch_texts, max_length, device)
        embeddings.append(batch_embeddings.detach().float().cpu())
    return torch.cat(embeddings, dim=0)


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer,
    output_dir: Path,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    train_state: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.encoder.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    write_json(
        output_dir / "embedding_config.json",
        {
            "base_model": args.model_name_or_path,
            "pooling": "last_token",
            "normalize": not args.no_normalize,
            "query_format": "{instruction}\\nQuery: {text}",
            "passage_format": "Represent this passage\\npassage: {text}",
            "append_eos": False,
            "padding_side": "left",
            "max_length": args.max_length,
            "loss": "qwen3_normalized_info_nce_with_row_aligned_hard_negative",
            "temperature": args.temperature,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": parse_lora_targets(args.lora_target_modules),
            "swanlab_project": args.swanlab_project,
            "eval_split": args.eval_split,
            "eval_metrics": ["ndcg@10", "recall@10"],
        },
    )
    if optimizer is not None:
        torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), output_dir / "scheduler.pt")
    if train_state is not None:
        state = dict(train_state)
        state["rng_state"] = collect_rng_state()
        torch.save(state, output_dir / "trainer_state.pt")
        write_json(
            output_dir / "trainer_state.json",
            {
                "global_step": state.get("global_step", 0),
                "epoch": state.get("epoch", 0),
                "next_batch_in_epoch": state.get("next_batch_in_epoch", 0),
                "total_steps": state.get("total_steps"),
                "completed": state.get("completed", False),
            },
        )


def build_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = (
        f"{timestamp}_qwen3embedding8b_lora-r{args.lora_r}_a{args.lora_alpha}"
        f"_b{args.per_device_batch_size}_ga{args.gradient_accumulation_steps}"
        f"_ep{args.num_train_epochs}_lr{args.learning_rate:g}_tau{args.temperature:g}_fp16"
    )
    return Path(args.output_root) / name


def parse_lora_targets(raw: str) -> list[str]:
    targets = [item.strip() for item in raw.split(",") if item.strip()]
    if not targets:
        raise ValueError("--lora_target_modules must contain at least one module name.")
    return targets


def resolve_resume_checkpoint(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"--resume_from_checkpoint does not exist or is not a directory: {path}")
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Checkpoint is missing adapter_config.json: {path}")
    if not (path / "trainer_state.pt").exists():
        raise FileNotFoundError(f"Checkpoint is missing trainer_state.pt: {path}")
    return path


def load_training_state(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    dist_info: dict,
) -> dict:
    optimizer_path = checkpoint_dir / "optimizer.pt"
    scheduler_path = checkpoint_dir / "scheduler.pt"
    state_path = checkpoint_dir / "trainer_state.pt"
    for path in (optimizer_path, scheduler_path, state_path):
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint is missing {path.name}: {checkpoint_dir}")

    map_location = device if device.type == "cuda" else torch.device("cpu")
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
    scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
    state = torch.load(state_path, map_location="cpu")
    restore_rng_state(state.get("rng_state"), dist_info)
    if dist_info["is_main"]:
        LOGGER.info("Loaded training state from %s", checkpoint_dir)
    return state


def collect_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None, dist_info: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if dist_info["is_main"]:
        LOGGER.info("Restored RNG state from checkpoint.")


def init_distributed() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = "LOCAL_RANK" in os.environ and world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
    return {
        "distributed": distributed,
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def enable_gradient_checkpointing(model: Qwen3EmbeddingModel) -> None:
    try:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.encoder.gradient_checkpointing_enable()
    if hasattr(model.encoder, "enable_input_require_grads"):
        model.encoder.enable_input_require_grads()


def log_trainable_parameters(model: torch.nn.Module) -> None:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    LOGGER.info("Trainable parameters: %s / %s (%.4f%%)", trainable, total, 100 * trainable / max(1, total))


def init_swanlab(args: argparse.Namespace, run_dir: Path, dist_info: dict) -> None:
    global SWANLAB_ACTIVE
    if args.disable_swanlab:
        LOGGER.info("SwanLab disabled by --disable_swanlab.")
        SWANLAB_ACTIVE = False
        return
    if swanlab is None:
        LOGGER.warning("SwanLab is not installed; skip SwanLab logging.")
        SWANLAB_ACTIVE = False
        return
    if not dist_info["is_main"]:
        return

    run_name = args.swanlab_run_name or run_dir.name
    project_name = sanitize_swanlab_project(args.swanlab_project)
    config = vars(args).copy()
    config.update(
        {
            "run_dir": str(run_dir),
            "world_size": dist_info["world_size"],
            "global_micro_batch_size": args.per_device_batch_size * dist_info["world_size"],
            "effective_batch_size": args.per_device_batch_size
            * args.gradient_accumulation_steps
            * dist_info["world_size"],
            "swanlab_requested_project": args.swanlab_project,
            "swanlab_actual_project": project_name,
        }
    )
    try:
        swanlab.init(project=project_name, name=run_name, config=config)
        SWANLAB_ACTIVE = True
    except Exception as exc:  # SwanLab connectivity should not kill training.
        LOGGER.warning("SwanLab init failed: %s", exc)
        SWANLAB_ACTIVE = False


def sanitize_swanlab_project(project_name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-+.")
    sanitized = "".join(ch if ch in allowed else "-" for ch in project_name.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "Contrastive-Model-fine-tune"


def log_swanlab(payload: dict[str, float], step: int) -> None:
    if swanlab is None or not SWANLAB_ACTIVE:
        return
    try:
        swanlab.log(payload, step=step)
    except Exception as exc:
        LOGGER.warning("SwanLab log failed at step=%s: %s", step, exc)


def finish_swanlab() -> None:
    global SWANLAB_ACTIVE
    if swanlab is None or not SWANLAB_ACTIVE:
        return
    try:
        swanlab.finish()
        SWANLAB_ACTIVE = False
    except Exception as exc:
        LOGGER.warning("SwanLab finish failed: %s", exc)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def setup_logging(is_main: bool) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if is_main else logging.WARNING,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
