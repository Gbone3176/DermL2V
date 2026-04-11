from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PYTHON_BIN = Path("/opt/conda/envs/l2v/bin/python")
BASE_MODEL_PATH = Path("/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
PEFT_MODEL_PATH = Path("/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291")
SUPERVISED_MODEL_PATH = Path("/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db")
RT_DATA_ROOT = Path("/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text")
RUN_ROOT = Path("/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k128_StructuredSelfAttn_gamma0p1_aux0p001")
OUTPUT_ROOT = Path("/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full")
SWEEP_DIR = OUTPUT_ROOT / "sweep_DermL2V_SM_SA_K128_gamma0p1_aux0p001_cp10to70"
SUMMARY_PATH = SWEEP_DIR / "summary.md"
LOG_DIR = SWEEP_DIR / "logs"
CUDA_DEVICE = os.environ.get("CUDA_DEVICE", "0")
MAX_LENGTH = os.environ.get("MAX_LENGTH", "512")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "64")
INSTRUCTION = "Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
POOLING_MODE = "structured_selfattn"

DATASETS = [
    ("DermSynth_knowledgebase", RT_DATA_ROOT / "eval3-text-benchmark_split_choices.jsonl"),
    ("MedMCQA_RT", RT_DATA_ROOT / "MedMCQA_RT_query_doc.jsonl"),
    ("MedQuAD_dermatology_qa_retrieval_doclt300", RT_DATA_ROOT / "MedQuAD_dermatology_qa_retrieval_doclt300.jsonl"),
]
SUMMARY_METRICS = ["NDCG@10", "Recall@10"]


def find_checkpoint_dir(step: int) -> Path:
    matches = sorted(RUN_ROOT.glob(f"**/checkpoint-{step}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one checkpoint-{step} under {RUN_ROOT}, found {len(matches)}")
    return matches[0]


def model_name(step: int) -> str:
    return f"DermL2V_SM_SA_K128_gamma0p1_aux0p001_cp{step}"


def result_path(dataset_name: str, step: int) -> Path:
    return SWEEP_DIR / dataset_name / f"{model_name(step)}.json"


def run_checkpoint(step: int) -> None:
    checkpoint_dir = find_checkpoint_dir(step)
    log_path = LOG_DIR / f"{model_name(step)}.log"

    with log_path.open("a", encoding="utf-8") as log_file:
        for dataset_name, dataset_path in DATASETS:
            output_file = result_path(dataset_name, step)
            if output_file.exists():
                continue

            cmd = [
                str(PYTHON_BIN),
                "-m",
                "experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_full",
                "--instruction",
                INSTRUCTION,
                "--dataset_file_path",
                str(dataset_path),
                "--model_name",
                model_name(step),
                "--pooling_mode",
                POOLING_MODE,
                "--max_length",
                str(MAX_LENGTH),
                "--batch_size",
                str(BATCH_SIZE),
                "--enable_bidirectional",
                "True",
                "--selfattn_attn_hidden_dim",
                "512",
                "--selfattn_num_hops",
                "8",
                "--selfattn_output_dropout",
                "0.0",
                "--selfattn_output_norm",
                "layernorm",
                "--base_model_name_or_path",
                str(BASE_MODEL_PATH),
                "--peft_model_name_or_path",
                str(PEFT_MODEL_PATH),
                "--extra_model_name_or_path",
                str(SUPERVISED_MODEL_PATH),
                str(checkpoint_dir),
                "--output",
                str(SWEEP_DIR),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
            subprocess.run(cmd, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def load_metrics(step: int) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for dataset_name, _ in DATASETS:
        path = result_path(dataset_name, step)
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        metrics[dataset_name] = json.loads(path.read_text())
    return metrics


def compute_avg_at10(metrics: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    avg_ndcg = sum(metrics[name]["NDCG@10"] for name, _ in DATASETS) / len(DATASETS)
    avg_recall = sum(metrics[name]["Recall@10"] for name, _ in DATASETS) / len(DATASETS)
    final_avg = (avg_ndcg + avg_recall) / 2.0
    return avg_ndcg, avg_recall, final_avg


def write_summary(rows: list[dict[str, object]], stopped_early: bool) -> None:
    lines = [
        "# Nonhomo Full Sweep Summary",
        "",
        f"Run root: `{RUN_ROOT}`",
        f"Output dir: `{SWEEP_DIR}`",
        "",
        "Metrics are reported as percentages.",
        "Avg means `(Avg NDCG@10 + Avg Recall@10) / 2` across the three nonhomo-full datasets.",
        "",
    ]
    if stopped_early:
        lines.append("Early stop triggered after two consecutive checkpoint-to-checkpoint drops in Avg.")
        lines.append("")

    header = ["CP"]
    for dataset_name, _ in DATASETS:
        for metric in SUMMARY_METRICS:
            header.append(f"{dataset_name} {metric} (%)")
    header.extend(["Avg NDCG@10 (%)", "Avg Recall@10 (%)", "Avg (%)"])

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:" for _ in header]) + "|")

    for row in rows:
        values = [str(row["cp"])]
        for dataset_name, _ in DATASETS:
            dataset_metrics = row["metrics"][dataset_name]
            for metric in SUMMARY_METRICS:
                values.append(f"{dataset_metrics[metric] * 100.0:.2f}")
        values.append(f"{row['avg_ndcg'] * 100.0:.2f}")
        values.append(f"{row['avg_recall'] * 100.0:.2f}")
        values.append(f"{row['avg'] * 100.0:.2f}")
        lines.append("| " + " | ".join(values) + " |")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    consecutive_drops = 0
    prev_avg: float | None = None
    stopped_early = False

    for step in range(10, 121, 10):
        run_checkpoint(step)
        metrics = load_metrics(step)
        avg_ndcg, avg_recall, final_avg = compute_avg_at10(metrics)
        rows.append(
            {
                "cp": step,
                "metrics": metrics,
                "avg_ndcg": avg_ndcg,
                "avg_recall": avg_recall,
                "avg": final_avg,
            }
        )
        write_summary(rows, stopped_early=False)

        if prev_avg is not None and final_avg < prev_avg:
            consecutive_drops += 1
        else:
            consecutive_drops = 0

        prev_avg = final_avg
        if consecutive_drops >= 2:
            stopped_early = True
            break

    write_summary(rows, stopped_early=stopped_early)


if __name__ == "__main__":
    main()
