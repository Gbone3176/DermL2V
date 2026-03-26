#!/usr/bin/env python3
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path("/storage/BioMedNLP/llm2vec")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASK_ORDER = ["SemVariants", "VisVariants", "DermQA", "SI1", "SI2"]
ROOT = REPO_ROOT / "experiments/deprecated/step_mix_probe_4gpu"
OUTPUT_DIR = ROOT / "outputs"
CONFIG_PATH = REPO_ROOT / "train_configs/supervised/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT.json"
SEPARATOR = "!@#$%^&*()"

DERM_EMBEDDING_PROMPTS = {
    "SemVariants": [
        "Read the provided dermatological condition description and return the candidate description that matches its meaning most closely.",
        "Given a skin-condition description, select the candidate description with the highest meaning-level similarity.",
        "Match the input dermatology description to the closest candidate description by semantics rather than exact wording.",
        "From all candidate descriptions, choose the one that best corresponds to the same dermatological condition described in the input.",
        "Identify the candidate description that is most conceptually aligned with the input skin-condition description.",
        "Return the single candidate description that best preserves the clinical meaning of the input dermatological description.",
        "Compare the input skin-condition description against the candidates and output the most semantically relevant one.",
        "Find the candidate description that would be the best paraphrase of the input dermatological condition description.",
        "Retrieve the candidate description that is nearest in meaning to the input dermatology description, ignoring surface-level phrasing.",
        "Select the candidate description that most accurately reflects the same underlying dermatological condition as the input description.",
    ],
    "VisVariants": [
        "Given a diagnosis-style dermatology text, retrieve the visual-description text that best matches it in meaning.",
        "Match a dermatological diagnosis or summary text to the most semantically aligned visual description text, and return the top match.",
        "Using the provided dermatology diagnostic statement as input, select the visual-description passage that is most relevant.",
        "From a pool of visual-description texts, return the one that most closely corresponds to the given dermatological diagnosis or summary.",
        "Identify the single visual-description text that best reflects the condition described by the provided diagnosis-oriented dermatology text.",
    ],
    "DermQA": [
        "Given a dermatology-related question, select the answer that is most relevant to what the question is asking.",
        "For a question in the skin-disease domain, return the candidate answer with the highest semantic relevance.",
        "Match the provided skin-disease question to the best corresponding answer from the available answers.",
        "From the answer candidates, pick the one that best aligns with the content and intent of the dermatology question.",
        "Given a skin-condition question, identify the answer that is most suitable and most closely related in meaning.",
        "Given a skin-disease question, retrieve the answer that is most semantically aligned, regardless of whether it concerns symptoms, diagnosis, or treatment.",
        "Select the answer that best matches the provided question within the dermatology domain.",
        "Given a dermatologic question, return the answer that most closely corresponds to the information being asked for.",
        "For the provided question about skin disorders, find and return the most relevant answer among the candidates.",
    ],
    "SI1": [
        "Retrieve the most appropriate answer for this dermatology question.",
        "Retrieve the answer entry that best matches the clinical vignette and question.",
        "Retrieve the most relevant response to the patient scenario described.",
        "Retrieve the best-matching answer based on the key clinical clues in the case.",
        "Given the dermatology prompt, retrieve the answer that most directly resolves the question.",
        "Retrieve the answer that is most consistent with standard dermatology clinical reasoning for this case.",
        "Retrieve the answer that best explains the findings and fits the question intent.",
        "Retrieve the answer that best aligns with the case details and the provided answer choices (if any).",
        "Retrieve the closest matching answer from the dataset for this dematology related query.",
    ],
    "SI2": [
        "Given a dermatology question, retrieve the single most relevant and most correct answer passage that directly answers it.",
        "Given a question about a skin condition, find the answer that most accurately and directly addresses the question.",
        "Find the dermatology answer passage that best matches this question and provides the highest-correctness response.",
        "Given a dermatologic presentation question, retrieve the answer that most directly answers it and is most medically accurate.",
        "Retrieve the most relevant dermatology answer that correctly resolves what the question is asking.",
        "Given a dermatology clinical question, retrieve the answer passage that is most relevant and has the highest factual correctness.",
        "Match this dermatology query to the answer that most precisely answers the question and is most correct.",
    ],
}


@dataclass
class ProbeSample:
    id_: int
    task_name: str
    query: str
    positive: str
    negative: str


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["per_device_train_batch_size"] = 64
    cfg["per_device_eval_batch_size"] = 64
    cfg["gradient_accumulation_steps"] = 8
    cfg["do_train"] = True
    cfg["num_processes"] = 4
    return cfg


def build_dataset(cfg: dict[str, Any]) -> list[ProbeSample]:
    random.seed(int(cfg.get("seed", 42)))
    file_path = cfg["dataset_file_path"]
    effective_batch_size = cfg["per_device_train_batch_size"] * cfg["num_processes"]
    dermqa_upsample_ratio = int(cfg.get("dermqa_upsample_ratio", 1))

    all_samples: list[ProbeSample] = []
    synthetic_id = 0

    for dataset_name in TASK_ORDER:
        dataset_path = os.path.join(file_path, f"{dataset_name}_train.jsonl")
        if not os.path.exists(dataset_path):
            continue
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_samples = [json.loads(line) for line in f if line.strip()]

        subtask_samples: list[ProbeSample] = []
        for sample in dataset_samples:
            instruction = random.choice(DERM_EMBEDDING_PROMPTS[dataset_name])
            query_text = sample["original"]
            pos_text = sample["positive_variant"]
            neg_text = sample["hard_negative_variant"]

            if dataset_name in {"SemVariants", "VisVariants", "SI1"}:
                query = f"{instruction}{SEPARATOR}{query_text}"
                pos = f"{instruction}{SEPARATOR}{pos_text}"
                neg = f"{instruction}{SEPARATOR}{neg_text}"
            else:
                query = f"{instruction}{SEPARATOR}{query_text}"
                pos = f"{SEPARATOR}{pos_text}"
                neg = f"{SEPARATOR}{neg_text}"

            subtask_samples.append(
                ProbeSample(
                    id_=synthetic_id,
                    task_name=dataset_name,
                    query=query,
                    positive=pos,
                    negative=neg,
                )
            )
            synthetic_id += 1

        if dataset_name == "DermQA" and dermqa_upsample_ratio > 1:
            for _ in range(dermqa_upsample_ratio):
                for sample in subtask_samples:
                    all_samples.append(
                        ProbeSample(
                            id_=synthetic_id,
                            task_name=sample.task_name,
                            query=sample.query,
                            positive=sample.positive,
                            negative=sample.negative,
                        )
                    )
                    synthetic_id += 1
        else:
            all_samples.extend(subtask_samples)

    indices = list(range(len(all_samples)))
    random.shuffle(indices)
    all_batches = []
    for i in range(0, len(indices), effective_batch_size):
        batch = indices[i : i + effective_batch_size]
        if len(batch) == effective_batch_size:
            all_batches.append(batch)
    random.shuffle(all_batches)

    final_idx_order = [idx for batch in all_batches for idx in batch]
    return [all_samples[idx] for idx in final_idx_order]


def summarize_steps(
    samples: list[dict[str, Any]],
    per_device_train_batch_size: int,
    num_processes: int,
    worker_name: str,
) -> dict[str, list[dict[str, Any]]]:
    global_batch_size = per_device_train_batch_size * num_processes
    step_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []

    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["step"], []).append(sample)

    for step in sorted(grouped):
        step_samples = sorted(grouped[step], key=lambda x: x["global_offset"])
        counts = Counter(item["task_name"] for item in step_samples)
        row = {
            "worker": worker_name,
            "step": step,
            "sample_count": len(step_samples),
        }
        for task in TASK_ORDER:
            row[f"{task}_count"] = counts.get(task, 0)
            row[f"{task}_ratio"] = counts.get(task, 0) / len(step_samples)
        step_rows.append(row)

        microstep_ids = sorted({item["microstep"] for item in step_samples})
        for microstep in microstep_ids:
            micro_samples = [item for item in step_samples if item["microstep"] == microstep]
            for rank in range(num_processes):
                rank_samples = [item for item in micro_samples if item["rank"] == rank]
                rank_counts = Counter(item["task_name"] for item in rank_samples)
                micro_row = {
                    "worker": worker_name,
                    "step": step,
                    "microstep": microstep,
                    "rank": rank,
                    "sample_count": len(rank_samples),
                    "global_batch_size": global_batch_size,
                }
                for task in TASK_ORDER:
                    micro_row[f"{task}_count"] = rank_counts.get(task, 0)
                    micro_row[f"{task}_ratio"] = rank_counts.get(task, 0) / len(rank_samples)
                micro_rows.append(micro_row)

    return {"step_rows": step_rows, "micro_rows": micro_rows}


def build_probe_records(
    cfg: dict[str, Any],
    dataset: list[ProbeSample],
    start_step: int,
    end_step: int,
    steps_per_epoch: int,
) -> list[dict[str, Any]]:
    per_device_train_batch_size = cfg["per_device_train_batch_size"]
    num_processes = cfg["num_processes"]
    grad_acc = cfg["gradient_accumulation_steps"]
    samples_per_step = per_device_train_batch_size * num_processes * grad_acc

    records: list[dict[str, Any]] = []
    for step in range(start_step, end_step + 1):
        epoch_index = (step - 1) // steps_per_epoch
        offset_in_epoch = ((step - 1) % steps_per_epoch) + 1
        start_index = (offset_in_epoch - 1) * samples_per_step
        end_index = offset_in_epoch * samples_per_step
        selected = dataset[start_index:end_index]
        for local_offset, sample in enumerate(selected):
            step_offset = local_offset
            global_offset = epoch_index * len(dataset) + start_index + local_offset
            microstep = step_offset // (per_device_train_batch_size * num_processes)
            micro_offset = step_offset % (per_device_train_batch_size * num_processes)
            rank = micro_offset // per_device_train_batch_size
            rank_offset = micro_offset % per_device_train_batch_size
            records.append(
                {
                    "step": step,
                    "epoch_index": epoch_index + 1,
                    "offset_in_epoch": offset_in_epoch,
                    "global_offset": global_offset,
                    "step_offset": step_offset,
                    "microstep": microstep,
                    "rank": rank,
                    "rank_offset": rank_offset,
                    "synthetic_id": sample.id_,
                    "task_name": sample.task_name,
                    "query": sample.query,
                    "positive": sample.positive,
                    "negative": sample.negative,
                }
            )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(
    cfg: dict[str, Any],
    step_df: pd.DataFrame,
    aggregate: dict[str, Any],
    start_step: int,
    end_step: int,
) -> str:
    lines = [
        "# Step 50-80 Dataset Mix Report",
        "",
        "## Settings",
        f"- config: `{CONFIG_PATH}`",
        f"- dataset: `{cfg['dataset_file_path']}`",
        f"- per_device_train_batch_size: `{cfg['per_device_train_batch_size']}`",
        f"- per_device_eval_batch_size: `{cfg['per_device_eval_batch_size']}`",
        f"- gradient_accumulation_steps: `{cfg['gradient_accumulation_steps']}`",
        f"- num_processes: `{cfg['num_processes']}`",
        f"- samples_per_optimizer_step: `{aggregate['samples_per_step']}`",
        f"- analyzed_steps: `{start_step}-{end_step}`",
        "",
        "## Aggregate Ratios",
    ]
    for task in TASK_ORDER:
        lines.append(
            f"- {task}: `{aggregate['task_counts'][task]}` samples, ratio `{aggregate['task_ratios'][task]:.6f}`"
        )

    lines.extend(
        [
            "",
            "## Per-Step Mean Ratios",
        ]
    )
    mean_ratios = step_df[[f"{task}_ratio" for task in TASK_ORDER]].mean().to_dict()
    for task in TASK_ORDER:
        lines.append(f"- {task}: `{mean_ratios[f'{task}_ratio']:.6f}`")

    lines.extend(
        [
            "",
            "## Notes",
            "- Step rows are true optimizer steps after `gradient_accumulation_steps=8`.",
            "- Each microstep contains one global batch of `64 x 4 = 256` samples.",
            "- Each rank receives a contiguous shard of `64` samples inside that microstep.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    dataset = build_dataset(cfg)

    start_step = 50
    end_step = 80
    samples_per_step = cfg["per_device_train_batch_size"] * cfg["num_processes"] * cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.floor(len(dataset) / samples_per_step)
    total_steps = steps_per_epoch * int(cfg.get("num_train_epochs", 1))

    if end_step > total_steps:
        raise ValueError(f"Requested end_step={end_step}, but dataset only yields {total_steps} full optimizer steps.")

    records = build_probe_records(
        cfg,
        dataset,
        start_step=start_step,
        end_step=end_step,
        steps_per_epoch=steps_per_epoch,
    )

    step_splits = [(50, 64, "worker_a"), (65, 80, "worker_b")]
    task_inputs = []
    for split_start, split_end, worker_name in step_splits:
        task_inputs.append(
            (
                [row for row in records if split_start <= row["step"] <= split_end],
                cfg["per_device_train_batch_size"],
                cfg["num_processes"],
                worker_name,
            )
        )

    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(summarize_steps, *args) for args in task_inputs]
        for future in futures:
            results.append(future.result())

    step_rows = [row for result in results for row in result["step_rows"]]
    micro_rows = [row for result in results for row in result["micro_rows"]]

    step_df = pd.DataFrame(step_rows).sort_values("step").reset_index(drop=True)
    micro_df = pd.DataFrame(micro_rows).sort_values(["step", "microstep", "rank"]).reset_index(drop=True)

    total_counts = Counter(row["task_name"] for row in records)
    aggregate = {
        "config_path": str(CONFIG_PATH),
        "dataset_file_path": cfg["dataset_file_path"],
        "start_step": start_step,
        "end_step": end_step,
        "num_processes": cfg["num_processes"],
        "steps_per_epoch": steps_per_epoch,
        "per_device_train_batch_size": cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "samples_per_step": samples_per_step,
        "total_selected_samples": len(records),
        "task_counts": {task: total_counts.get(task, 0) for task in TASK_ORDER},
        "task_ratios": {
            task: total_counts.get(task, 0) / len(records)
            for task in TASK_ORDER
        },
    }

    summary_path = OUTPUT_DIR / "step_50_80_summary.csv"
    micro_path = OUTPUT_DIR / "step_50_80_rank_microbatch_summary.csv"
    samples_path = OUTPUT_DIR / "step_50_80_samples.jsonl"
    aggregate_path = OUTPUT_DIR / "step_50_80_aggregate.json"
    report_path = OUTPUT_DIR / "step_50_80_report.md"

    step_df.to_csv(summary_path, index=False)
    micro_df.to_csv(micro_path, index=False)
    write_jsonl(samples_path, records)
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    report_path.write_text(build_report(cfg, step_df, aggregate, start_step, end_step), encoding="utf-8")

    print(f"Loaded train samples: {len(dataset)}")
    print(f"Total optimizer steps available: {total_steps}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {micro_path}")
    print(f"Wrote: {samples_path}")
    print(f"Wrote: {aggregate_path}")
    print(f"Wrote: {report_path}")
    print("Aggregate ratios:")
    for task in TASK_ORDER:
        print(f"  {task}: {aggregate['task_counts'][task]} ({aggregate['task_ratios'][task]:.6f})")


if __name__ == "__main__":
    main()
