#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_ORDER = ["SemVariants", "VisVariants", "DermQA", "SI1", "SI2"]
PROMPT_COUNTS = {
    "SemVariants": 10,
    "VisVariants": 5,
    "DermQA": 9,
    "SI1": 9,
    "SI2": 7,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze step-level task mix, margin, and grad_norm around epoch boundaries."
    )
    parser.add_argument("config", type=Path, help="Training config JSON path.")
    parser.add_argument("trainer_state", type=Path, help="trainer_state.json path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/retrospectives/data/step_dynamics_outputs"),
        help="Directory to write CSV and plots.",
    )
    parser.add_argument(
        "--focus-start",
        type=int,
        default=45,
        help="Focus window start step for reporting.",
    )
    parser.add_argument(
        "--focus-end",
        type=int,
        default=95,
        help="Focus window end step for reporting.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rebuild_train_sequence(cfg: dict):
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    base = Path(cfg["dataset_file_path"])
    dermqa_upsample_ratio = int(cfg.get("dermqa_upsample_ratio", 1))
    per_device_train_batch_size = int(cfg["per_device_train_batch_size"])
    num_processes = 4
    effective_batch_size = per_device_train_batch_size * num_processes

    samples = []
    synthetic_id = 0

    for dataset in DATASET_ORDER:
        path = base / f"{dataset}_train.jsonl"
        rows = load_jsonl(path)
        prompt_pool = list(range(PROMPT_COUNTS[dataset]))
        subtask = []
        for row in rows:
            _ = random.choice(prompt_pool)
            margin = float(row["pos_score"]) - float(row["neg_score"])
            subtask.append(
                {
                    "synthetic_id": synthetic_id,
                    "task": dataset,
                    "source_id": row["id"],
                    "margin": margin,
                }
            )
            synthetic_id += 1

        if dataset == "DermQA" and dermqa_upsample_ratio > 1:
            for _ in range(dermqa_upsample_ratio):
                for sample in subtask:
                    samples.append(
                        {
                            "synthetic_id": synthetic_id,
                            "task": sample["task"],
                            "source_id": sample["source_id"],
                            "margin": sample["margin"],
                        }
                    )
                    synthetic_id += 1
        else:
            samples.extend(subtask)

    indices = list(range(len(samples)))
    random.shuffle(indices)

    all_batches = []
    for i in range(0, len(indices), effective_batch_size):
        batch = indices[i : i + effective_batch_size]
        if len(batch) == effective_batch_size:
            all_batches.append(batch)

    random.shuffle(all_batches)
    final_sequence = [samples[idx] for batch in all_batches for idx in batch]
    return final_sequence


def build_step_dataframe(cfg: dict, final_sequence):
    per_device_train_batch_size = int(cfg["per_device_train_batch_size"])
    gradient_accumulation_steps = int(cfg["gradient_accumulation_steps"])
    num_processes = 4
    samples_per_step = per_device_train_batch_size * num_processes * gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(final_sequence) / samples_per_step)

    rows = []
    max_steps = math.ceil(float(cfg.get("num_train_epochs", 1)) * steps_per_epoch)
    for step in range(1, max_steps + 1):
        epoch_idx = (step - 1) // steps_per_epoch
        offset_in_epoch = (step - 1) % steps_per_epoch + 1
        start = (offset_in_epoch - 1) * samples_per_step
        end = min(offset_in_epoch * samples_per_step, len(final_sequence))
        chunk = final_sequence[start:end]
        if not chunk:
            continue
        counts = Counter(sample["task"] for sample in chunk)
        row = {
            "step": step,
            "epoch_index": epoch_idx + 1,
            "offset_in_epoch": offset_in_epoch,
            "samples_in_step": len(chunk),
            "mean_margin": sum(sample["margin"] for sample in chunk) / len(chunk),
            "epoch_boundary_step": offset_in_epoch == 1,
        }
        for task in DATASET_ORDER:
            row[f"{task}_count"] = counts.get(task, 0)
            row[f"{task}_ratio"] = counts.get(task, 0) / len(chunk)
        rows.append(row)

    return pd.DataFrame(rows), steps_per_epoch


def build_train_log_dataframe(trainer_state: dict):
    rows = []
    for item in trainer_state.get("log_history", []):
        if "step" not in item:
            continue
        rows.append(
            {
                "step": int(item["step"]),
                "loss": item.get("loss"),
                "grad_norm": item.get("grad_norm"),
                "learning_rate": item.get("learning_rate"),
                "log_epoch": item.get("epoch"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["step", "loss", "grad_norm", "learning_rate", "log_epoch"])
    return pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last")


def save_focus_report(df: pd.DataFrame, focus_start: int, focus_end: int, out_path: Path):
    cols = [
        "step",
        "epoch_index",
        "offset_in_epoch",
        "samples_in_step",
        "mean_margin",
        "grad_norm",
        "loss",
        "learning_rate",
    ] + [f"{task}_ratio" for task in DATASET_ORDER]
    focus = df[(df["step"] >= focus_start) & (df["step"] <= focus_end)][cols].copy()
    focus.to_csv(out_path, index=False)
    return focus


def make_plot(df: pd.DataFrame, steps_per_epoch: int, out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    axes[0].plot(df["step"], df["grad_norm"], marker="o", label="grad_norm")
    axes[0].set_ylabel("grad_norm")
    axes[0].set_title("Step Dynamics")
    axes[0].legend(loc="upper right")

    axes[1].plot(df["step"], df["mean_margin"], marker="o", color="tab:orange", label="mean margin")
    axes[1].set_ylabel("mean pos-neg margin")
    axes[1].legend(loc="upper right")

    for task in DATASET_ORDER:
        axes[2].plot(df["step"], df[f"{task}_ratio"], marker="o", label=task)
    axes[2].set_ylabel("task ratio")
    axes[2].set_xlabel("step")
    axes[2].legend(loc="upper right", ncol=3)

    max_step = int(df["step"].max())
    boundary = steps_per_epoch
    while boundary <= max_step:
        for ax in axes:
            ax.axvline(boundary, color="red", linestyle="--", linewidth=1, alpha=0.6)
        boundary += steps_per_epoch

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main():
    args = parse_args()
    cfg = load_json(args.config)
    trainer_state = load_json(args.trainer_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_sequence = rebuild_train_sequence(cfg)
    step_df, steps_per_epoch = build_step_dataframe(cfg, final_sequence)
    log_df = build_train_log_dataframe(trainer_state)
    merged = step_df.merge(log_df, on="step", how="left")

    merged_path = args.output_dir / "step_dynamics_full.csv"
    merged.to_csv(merged_path, index=False)

    focus_path = args.output_dir / f"focus_steps_{args.focus_start}_{args.focus_end}.csv"
    focus = save_focus_report(merged, args.focus_start, args.focus_end, focus_path)

    plot_path = args.output_dir / "step_dynamics_overview.png"
    make_plot(merged[merged["grad_norm"].notna()].copy(), steps_per_epoch, plot_path)

    print(f"steps_per_epoch={steps_per_epoch}")
    print(f"wrote: {merged_path}")
    print(f"wrote: {focus_path}")
    print(f"wrote: {plot_path}")
    print()
    print("Focus window summary:")
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
