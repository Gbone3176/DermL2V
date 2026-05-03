#!/usr/bin/env python
"""Fine-tune PubMedBERT embeddings on DermVariants triplets."""

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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

try:
    import swanlab
except ImportError:  # pragma: no cover - optional runtime dependency
    swanlab = None

THIS_FILE = Path(__file__).resolve()
CONTRASTIVE_ROOT = THIS_FILE.parents[1]
if str(CONTRASTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRASTIVE_ROOT))

from shared.dermvariants import DermTripletDataset, collate_triplets, load_dermvariants_triplets
from shared.losses import RowAlignedHardNegativeNLLLoss
from shared.modeling import BertEmbeddingModel


LOGGER = logging.getLogger("train_pubmedbert")
SWANLAB_ACTIVE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--data_dir", default="/storage/dataset/dermatoscop/Derm1M/DermVariantsData")
    parser.add_argument("--output_root", default="ContrastiveModel/PubMedBERT/output")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--loss_scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dermqa_upsample_ratio", type=int, default=1)
    parser.add_argument("--separator", default="!@#$%^&*()")
    parser.add_argument("--no_query_instruction", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--swanlab_project", default="Contrastive Model fine-tune")
    parser.add_argument("--swanlab_run_name", default=None)
    parser.add_argument("--disable_swanlab", action="store_true")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--eval_include_negatives", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist_info = init_distributed()
    setup_logging(dist_info["is_main"])
    set_seed(args.seed + dist_info["rank"])

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")

    device = torch.device("cuda", dist_info["local_rank"]) if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.info("Using device=%s rank=%s world_size=%s", device, dist_info["rank"], dist_info["world_size"])

    samples = load_dermvariants_triplets(
        data_dir=args.data_dir,
        split=args.split,
        effective_batch_size=args.per_device_batch_size * max(1, dist_info["world_size"]),
        seed=args.seed,
        separator=args.separator,
        dermqa_upsample_ratio=args.dermqa_upsample_ratio,
        use_query_instruction=not args.no_query_instruction,
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
        raise ValueError(
            "No full training batch is available. Lower --per_device_batch_size "
            "or increase --max_train_samples / dataset size."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    model = BertEmbeddingModel(
        model_name_or_path=args.model_name_or_path,
        pooling=args.pooling,
        normalize=True,
        local_files_only=args.local_files_only,
    ).to(device)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    if dist_info["distributed"]:
        model = DistributedDataParallel(model, device_ids=[dist_info["local_rank"]] if torch.cuda.is_available() else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(dataloader) // args.gradient_accumulation_steps)
    total_steps = args.max_steps if args.max_steps > 0 else max(1, math.ceil(args.num_train_epochs * steps_per_epoch))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = RowAlignedHardNegativeNLLLoss(scale=args.loss_scale)

    run_dir = build_run_dir(args)
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
            },
        )
        eval_samples = load_dermvariants_triplets(
            data_dir=args.data_dir,
            split=args.eval_split,
            effective_batch_size=1,
            shuffle_individual_datasets=False,
            seed=args.seed,
            separator=args.separator,
            dermqa_upsample_ratio=1,
            use_query_instruction=not args.no_query_instruction,
            max_samples=args.eval_max_samples,
        )
        LOGGER.info("Loaded %s eval triplets from split=%s", len(eval_samples), args.eval_split)
        init_swanlab(args, run_dir, dist_info)

    eval_steps = args.eval_every_steps if args.eval_every_steps > 0 else steps_per_epoch

    train(
        args,
        model,
        tokenizer,
        dataloader,
        sampler,
        optimizer,
        scheduler,
        loss_fn,
        device,
        total_steps,
        dist_info,
        run_dir,
        eval_samples,
        eval_steps,
    )

    if dist_info["is_main"]:
        save_checkpoint(model, tokenizer, run_dir / "final", args)
        LOGGER.info("Saved final checkpoint to %s", run_dir / "final")
        finish_swanlab()

    cleanup_distributed()


def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer,
    dataloader: DataLoader,
    sampler,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: RowAlignedHardNegativeNLLLoss,
    device: torch.device,
    total_steps: int,
    dist_info: dict,
    run_dir: Path,
    eval_samples: list | None,
    eval_steps: int,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running_loss = 0.0
    epoch = 0
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None
    progress = tqdm(total=total_steps, disable=not dist_info["is_main"], desc="training")

    while global_step < total_steps:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            should_step = (step + 1) % args.gradient_accumulation_steps == 0
            sync_context = (
                model.no_sync()
                if dist_info["distributed"] and not should_step and hasattr(model, "no_sync")
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None and device.type == "cuda",
                ):
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
            progress.update(1)

            if dist_info["is_main"] and global_step % args.logging_steps == 0:
                avg_loss = running_loss / max(1, args.logging_steps)
                LOGGER.info("step=%s loss=%.6f lr=%.6g", global_step, avg_loss, scheduler.get_last_lr()[0])
                log_swanlab({"train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                running_loss = 0.0

            if dist_info["is_main"] and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, run_dir / f"checkpoint-{global_step}", args)

            if eval_steps > 0 and (global_step % eval_steps == 0 or global_step >= total_steps):
                run_validation_eval(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    eval_samples=eval_samples,
                    device=device,
                    global_step=global_step,
                    dist_info=dist_info,
                    run_dir=run_dir,
                )

            if global_step >= total_steps:
                break
        epoch += 1

    progress.close()


def encode_triplets(
    model,
    tokenizer,
    batch: dict[str, list[str]],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch["query"])
    texts = batch["query"] + batch["positive"] + batch["negative"]
    embeddings = encode_texts(model, tokenizer, texts, max_length, device)
    query = embeddings[:batch_size]
    positive = embeddings[batch_size : 2 * batch_size]
    negative = embeddings[2 * batch_size :]
    return query, positive, negative


def encode_texts(model, tokenizer, texts: list[str], max_length: int, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return model(**encoded)


def run_validation_eval(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer,
    eval_samples: list | None,
    device: torch.device,
    global_step: int,
    dist_info: dict,
    run_dir: Path,
) -> None:
    if dist_info["is_main"]:
        if not eval_samples:
            LOGGER.warning("Skip eval at step=%s because eval_samples is empty.", global_step)
        else:
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
            LOGGER.info(
                "eval step=%s ndcg@10=%.6f recall@10=%.6f",
                global_step,
                metrics["ndcg@10"],
                metrics["recall@10"],
            )
            log_swanlab(
                {
                    "eval/ndcg@10": metrics["ndcg@10"],
                    "eval/recall@10": metrics["recall@10"],
                },
                step=global_step,
            )

    if dist_info["distributed"]:
        dist.barrier()


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

    query_texts = [sample.query for sample in samples]
    doc_texts = [sample.positive for sample in samples]
    relevant_doc_indices = list(range(len(doc_texts)))

    if include_negatives:
        doc_texts.extend(sample.negative for sample in samples)

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
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_embeddings = encode_texts(model, tokenizer, batch_texts, max_length, device)
        embeddings.append(F.normalize(batch_embeddings.detach().float().cpu(), p=2, dim=-1))
    return torch.cat(embeddings, dim=0)


def save_checkpoint(model: torch.nn.Module, tokenizer, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_json(
        output_dir / "embedding_config.json",
        {
            "base_model": args.model_name_or_path,
            "pooling": args.pooling,
            "normalize": True,
            "max_length": args.max_length,
            "loss": "row_aligned_hard_negative_nll",
            "loss_scale": args.loss_scale,
            "swanlab_project": args.swanlab_project,
            "eval_split": args.eval_split,
            "eval_metrics": ["ndcg@10", "recall@10"],
        },
    )


def build_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_slug = "pubmedbert-base-embeddings"
    name = (
        f"{timestamp}_{model_slug}_pool-{args.pooling}_b{args.per_device_batch_size}"
        f"_ga{args.gradient_accumulation_steps}_ep{args.num_train_epochs}"
        f"_lr{args.learning_rate:g}_scale{args.loss_scale:g}"
    )
    return Path(args.output_root) / name


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


def enable_gradient_checkpointing(model: BertEmbeddingModel) -> None:
    try:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.encoder.gradient_checkpointing_enable()


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
    if project_name != args.swanlab_project:
        LOGGER.info(
            "SwanLab project name sanitized from %r to %r.",
            args.swanlab_project,
            project_name,
        )
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
        swanlab.init(
            project=project_name,
            name=run_name,
            config=config,
        )
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
