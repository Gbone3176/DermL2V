import argparse
import json
import math
import os
import random
from copy import deepcopy

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import DermVariantsTripletDataset, TripletCollator
from .losses import supervised_contrastive_loss
from .model import (
    attach_lora_to_text_encoder,
    encode_batch,
    enable_gradient_checkpointing,
    load_nvembed_model,
    move_batch_to_device,
    save_training_artifacts,
    trainable_parameter_summary,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--train_file_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--separator", type=str, default="!@#$%^&*()")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--dermqa_upsample_ratio", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        parser.set_defaults(**config_data)
        args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, data_loader, device, temperature: float):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        query_embeds = encode_batch(model, batch["query"])
        positive_embeds = encode_batch(model, batch["positive"])
        negative_embeds = encode_batch(model, batch["negative"])
        loss = supervised_contrastive_loss(query_embeds, positive_embeds, negative_embeds, temperature)
        total_loss += loss.item()
        total_steps += 1
    model.train()
    return total_loss / max(total_steps, 1)


def make_dataloaders(args, tokenizer):
    train_dataset = DermVariantsTripletDataset(
        file_path=args.train_file_path,
        split="train",
        separator=args.separator,
        dermqa_upsample_ratio=args.dermqa_upsample_ratio,
        seed=args.seed,
    )
    train_collator = TripletCollator(tokenizer, max_length=args.max_length, separator=args.separator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_collator,
    )

    val_loader = None
    has_validation = any(
        os.path.exists(os.path.join(args.train_file_path, f"{task_name}_validation.jsonl"))
        for task_name in ["SemVariants", "VisVariants", "DermQA", "SI1", "SI2"]
    )
    if has_validation:
        val_dataset = DermVariantsTripletDataset(
            file_path=args.train_file_path,
            split="validation",
            separator=args.separator,
            dermqa_upsample_ratio=1,
            seed=args.seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=train_collator,
        )
    return train_loader, val_loader


def main():
    args = parse_args()
    if args.model_name_or_path is None or args.train_file_path is None or args.output_dir is None:
        raise ValueError("model_name_or_path, train_file_path, and output_dir are required")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")
    device = torch.device("cuda")

    model, tokenizer = load_nvembed_model(
        args.model_name_or_path,
        fp16=args.fp16,
        local_files_only=args.local_files_only,
    )
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model.embedding_model)
    model = attach_lora_to_text_encoder(
        model,
        target_modules=args.lora_target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)
    model.train()

    print(trainable_parameter_summary(model))

    train_loader, val_loader = make_dataloaders(args, tokenizer)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_optimization_steps = max(updates_per_epoch * args.num_train_epochs, 1)
    warmup_steps = int(total_optimization_steps * args.warmup_ratio)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimization_steps,
    )
    scaler = GradScaler(enabled=args.fp16)

    global_step = 0
    best_val_loss = None
    progress = tqdm(total=total_optimization_steps, desc="train")

    for epoch in range(args.num_train_epochs):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            query_embeds = encode_batch(model, batch["query"])
            positive_embeds = encode_batch(model, batch["positive"])
            negative_embeds = encode_batch(model, batch["negative"])
            loss = supervised_contrastive_loss(query_embeds, positive_embeds, negative_embeds, args.temperature)
            loss = loss / args.gradient_accumulation_steps
            running_loss += loss.item()

            scaler.scale(loss).backward()

            if step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                progress.update(1)

                if global_step % args.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    progress.set_postfix(loss=f"{running_loss:.4f}", lr=f"{current_lr:.2e}")
                    running_loss = 0.0

                if val_loader is not None and global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device, args.temperature)
                    print(f"step={global_step} val_loss={val_loss:.6f}")
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_training_artifacts(
                            model,
                            tokenizer,
                            os.path.join(args.output_dir, "best_checkpoint"),
                            deepcopy(vars(args)),
                            {"global_step": global_step, "epoch": epoch, "best_val_loss": best_val_loss},
                        )

                if global_step % args.save_steps == 0:
                    save_training_artifacts(
                        model,
                        tokenizer,
                        os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                        deepcopy(vars(args)),
                        {"global_step": global_step, "epoch": epoch, "best_val_loss": best_val_loss},
                    )

        if len(train_loader) % args.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            progress.update(1)

    progress.close()
    final_val_loss = evaluate(model, val_loader, device, args.temperature) if val_loader is not None else None
    save_training_artifacts(
        model,
        tokenizer,
        os.path.join(args.output_dir, "final_checkpoint"),
        deepcopy(vars(args)),
        {"global_step": global_step, "final_val_loss": final_val_loss},
    )
    print(f"finished global_step={global_step} final_val_loss={final_val_loss}")
