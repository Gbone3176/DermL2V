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
RUN_ROOT = Path("/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k16_StructuredSelfAttn_gamma0p001_aux0p001/DermVariants_train_m-Meta-Llama-31-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16")
OUTPUT_DIR = Path("/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full/sweep_DermL2V_SM_SA_K16_gamma0p001_aux0p001_cp30to70")
LOG_DIR = OUTPUT_DIR / "logs"
SUMMARY_AT10 = OUTPUT_DIR / "summary_at10.md"
INSTRUCTION = "Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
POOLING_MODE = "structured_selfattn"
MAX_LENGTH = os.environ.get("MAX_LENGTH", "512")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "64")
TARGET_STEPS = [30, 40, 50, 60, 70]
MAX_GPUS = 5

DATASETS = [
    ("DermSynth_knowledgebase", RT_DATA_ROOT / "eval3-text-benchmark_split_choices.jsonl"),
    ("MedMCQA_RT", RT_DATA_ROOT / "MedMCQA_RT_query_doc.jsonl"),
    ("MedQuAD_dermatology_qa_retrieval_doclt300", RT_DATA_ROOT / "MedQuAD_dermatology_qa_retrieval_doclt300.jsonl"),
]


def detect_free_gpus(max_memory_used_mb: int = 1024) -> list[int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = []
    for line in out.strip().splitlines():
        gpu_str, mem_str = [part.strip() for part in line.split(",")]
        if int(mem_str) <= max_memory_used_mb:
            free.append(int(gpu_str))
    return free


def find_checkpoint_dir(step: int) -> Path:
    matches = sorted(RUN_ROOT.glob(f"**/checkpoint-{step}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one checkpoint-{step} under {RUN_ROOT}, found {len(matches)}")
    return matches[0]


def model_name(step: int) -> str:
    return f"DermL2V_SM_SA_K16_gamma0p001_aux0p001_cp{step}"


def result_path(dataset_name: str, step: int) -> Path:
    return OUTPUT_DIR / dataset_name / f"{model_name(step)}.json"


def checkpoint_complete(step: int) -> bool:
    return all(result_path(dataset_name, step).exists() for dataset_name, _ in DATASETS)


def build_eval_command(step: int) -> str:
    checkpoint_dir = find_checkpoint_dir(step)
    log_path = LOG_DIR / f"{model_name(step)}.log"
    command_lines = []
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
            str(OUTPUT_DIR),
        ]
        command_lines.append(" ".join(subprocess.list2cmdline([part]) for part in cmd))
    if not command_lines:
        return f"echo '{model_name(step)} already complete' >> {subprocess.list2cmdline([str(log_path)])}"
    return " && ".join(f"{line} >> {subprocess.list2cmdline([str(log_path)])} 2>&1" for line in command_lines)


def load_metrics(step: int) -> dict[str, dict[str, float]]:
    return {
        dataset_name: json.loads(result_path(dataset_name, step).read_text())
        for dataset_name, _ in DATASETS
    }


def write_summary_at10() -> None:
    rows = []
    for step in TARGET_STEPS:
        if not checkpoint_complete(step):
            continue
        metrics = load_metrics(step)
        avg_ndcg = sum(metrics[name]["NDCG@10"] for name, _ in DATASETS) / len(DATASETS)
        avg_recall = sum(metrics[name]["Recall@10"] for name, _ in DATASETS) / len(DATASETS)
        avg = (avg_ndcg + avg_recall) / 2.0
        rows.append((step, metrics, avg_ndcg, avg_recall, avg))

    lines = [
        "# Nonhomo Full Parallel Sweep Summary at @10",
        "",
        f"Run root: `{RUN_ROOT}`",
        f"Output dir: `{OUTPUT_DIR}`",
        "",
        "Metrics are reported as percentages.",
        "",
        "| CP | DermSynth NDCG@10 (%) | DermSynth Recall@10 (%) | MedMCQA NDCG@10 (%) | MedMCQA Recall@10 (%) | MedQuAD-doclt300 NDCG@10 (%) | MedQuAD-doclt300 Recall@10 (%) | Avg NDCG@10 (%) | Avg Recall@10 (%) | Avg (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step, metrics, avg_ndcg, avg_recall, avg in rows:
        lines.append(
            "| {step} | {d_ndcg:.2f} | {d_recall:.2f} | {m_ndcg:.2f} | {m_recall:.2f} | {q_ndcg:.2f} | {q_recall:.2f} | {avg_ndcg:.2f} | {avg_recall:.2f} | {avg:.2f} |".format(
                step=step,
                d_ndcg=metrics["DermSynth_knowledgebase"]["NDCG@10"] * 100.0,
                d_recall=metrics["DermSynth_knowledgebase"]["Recall@10"] * 100.0,
                m_ndcg=metrics["MedMCQA_RT"]["NDCG@10"] * 100.0,
                m_recall=metrics["MedMCQA_RT"]["Recall@10"] * 100.0,
                q_ndcg=metrics["MedQuAD_dermatology_qa_retrieval_doclt300"]["NDCG@10"] * 100.0,
                q_recall=metrics["MedQuAD_dermatology_qa_retrieval_doclt300"]["Recall@10"] * 100.0,
                avg_ndcg=avg_ndcg * 100.0,
                avg_recall=avg_recall * 100.0,
                avg=avg * 100.0,
            )
        )
    SUMMARY_AT10.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    free_gpus = detect_free_gpus()
    if len(free_gpus) < MAX_GPUS:
        raise RuntimeError(f"Need at least {MAX_GPUS} free GPUs, found {len(free_gpus)}: {free_gpus}")

    steps_to_run = [step for step in TARGET_STEPS if not checkpoint_complete(step)]
    if not steps_to_run:
        write_summary_at10()
        print("All target checkpoints already complete.")
        return

    if len(steps_to_run) > MAX_GPUS:
        raise RuntimeError(f"Configured for at most {MAX_GPUS} parallel checkpoints, got {len(steps_to_run)}")

    selected_gpus = free_gpus[:MAX_GPUS][: len(steps_to_run)]
    processes = []
    for gpu_id, step in zip(selected_gpus, steps_to_run):
        cmd = build_eval_command(step)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-lc", cmd],
            env=env,
            cwd="/storage/BioMedNLP/llm2vec",
        )
        processes.append((step, gpu_id, proc))
        print(f"Started cp{step} on GPU {gpu_id}")

    failed = []
    for step, gpu_id, proc in processes:
        return_code = proc.wait()
        print(f"Finished cp{step} on GPU {gpu_id} with code {return_code}")
        if return_code != 0:
            failed.append((step, gpu_id, return_code))

    write_summary_at10()
    if failed:
        raise RuntimeError(f"Some checkpoint jobs failed: {failed}")


if __name__ == "__main__":
    main()
