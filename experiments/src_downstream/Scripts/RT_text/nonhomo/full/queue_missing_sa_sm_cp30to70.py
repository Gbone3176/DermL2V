from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


PYTHON_BIN = Path("/opt/conda/envs/l2v/bin/python")
BASE_MODEL_PATH = Path("/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
PEFT_MODEL_PATH = Path("/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291")
SUPERVISED_MODEL_PATH = Path("/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db")
RT_DATA_ROOT = Path("/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text")
OUTPUT_ROOT = Path("/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full")
MAX_LENGTH = os.environ.get("MAX_LENGTH", "512")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "64")
TARGET_STEPS = [30, 40, 50, 60, 70]
MAX_GPUS = 5
INSTRUCTION = "Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."

DATASETS = [
    ("DermSynth_knowledgebase", RT_DATA_ROOT / "eval3-text-benchmark_split_choices.jsonl"),
    ("MedMCQA_RT", RT_DATA_ROOT / "MedMCQA_RT_query_doc.jsonl"),
    ("MedQuAD_dermatology_qa_retrieval_doclt300", RT_DATA_ROOT / "MedQuAD_dermatology_qa_retrieval_doclt300.jsonl"),
]

EXPERIMENTS = [
    {
        "run_root": Path("/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k8_StructuredSelfAttn_gamma0p001_aux0p001/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16"),
        "output_dir": OUTPUT_ROOT / "sweep_DermL2V_SM_SA_K8_gamma0p001_aux0p001_cp30to70",
        "model_prefix": "DermL2V_SM_SA_K8_gamma0p001_aux0p001",
    },
    {
        "run_root": Path("/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k32_StructuredSelfAttn_gamma0p001_aux0p001/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16"),
        "output_dir": OUTPUT_ROOT / "sweep_DermL2V_SM_SA_K32_gamma0p001_aux0p001_cp30to70",
        "model_prefix": "DermL2V_SM_SA_K32_gamma0p001_aux0p001",
    },
    {
        "run_root": Path("/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k64_StructuredSelfAttn_gamma0p001_aux0p001/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16"),
        "output_dir": OUTPUT_ROOT / "sweep_DermL2V_SM_SA_K64_gamma0p001_aux0p001_cp30to70",
        "model_prefix": "DermL2V_SM_SA_K64_gamma0p001_aux0p001",
    },
    {
        "run_root": Path("/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k128_StructuredSelfAttn_gamma0p1_aux0p01/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16"),
        "output_dir": OUTPUT_ROOT / "sweep_DermL2V_SM_SA_K128_gamma0p1_aux0p01_cp30to70",
        "model_prefix": "DermL2V_SM_SA_K128_gamma0p1_aux0p01",
    },
    {
        "run_root": Path("/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k128_StructuredSelfAttn_gamma0p1_aux0p1/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16"),
        "output_dir": OUTPUT_ROOT / "sweep_DermL2V_SM_SA_K128_gamma0p1_aux0p1_cp30to70",
        "model_prefix": "DermL2V_SM_SA_K128_gamma0p1_aux0p1",
    },
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


def wait_for_free_gpus(required: int = MAX_GPUS, poll_seconds: int = 60) -> list[int]:
    while True:
        free = detect_free_gpus()
        if len(free) >= required:
            return free[:required]
        print(f"Waiting for {required} free GPUs, found {len(free)}: {free}")
        time.sleep(poll_seconds)


def find_checkpoint_dir(run_root: Path, step: int) -> Path:
    matches = sorted(run_root.glob(f"**/checkpoint-{step}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one checkpoint-{step} under {run_root}, found {len(matches)}")
    return matches[0]


def model_name(model_prefix: str, step: int) -> str:
    return f"{model_prefix}_cp{step}"


def result_path(output_dir: Path, dataset_name: str, model_prefix: str, step: int) -> Path:
    return output_dir / dataset_name / f"{model_name(model_prefix, step)}.json"


def checkpoint_complete(output_dir: Path, model_prefix: str, step: int) -> bool:
    return all(result_path(output_dir, dataset_name, model_prefix, step).exists() for dataset_name, _ in DATASETS)


def experiment_complete(output_dir: Path, model_prefix: str) -> bool:
    return all(checkpoint_complete(output_dir, model_prefix, step) for step in TARGET_STEPS)


def build_eval_command(run_root: Path, output_dir: Path, model_prefix: str, step: int) -> str:
    checkpoint_dir = find_checkpoint_dir(run_root, step)
    log_dir = output_dir / "logs"
    log_path = log_dir / f"{model_name(model_prefix, step)}.log"
    command_lines = []
    for dataset_name, dataset_path in DATASETS:
        output_file = result_path(output_dir, dataset_name, model_prefix, step)
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
            model_name(model_prefix, step),
            "--pooling_mode",
            "structured_selfattn",
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
            str(output_dir),
        ]
        command_lines.append(" ".join(subprocess.list2cmdline([part]) for part in cmd))
    if not command_lines:
        return f"echo '{model_name(model_prefix, step)} already complete' >> {subprocess.list2cmdline([str(log_path)])}"
    return " && ".join(f"{line} >> {subprocess.list2cmdline([str(log_path)])} 2>&1" for line in command_lines)


def write_summary_at10(run_root: Path, output_dir: Path, model_prefix: str) -> None:
    rows = []
    for step in TARGET_STEPS:
        if not checkpoint_complete(output_dir, model_prefix, step):
            continue
        metrics = {
            dataset_name: json.loads(result_path(output_dir, dataset_name, model_prefix, step).read_text())
            for dataset_name, _ in DATASETS
        }
        avg_ndcg = sum(metrics[name]["NDCG@10"] for name, _ in DATASETS) / len(DATASETS)
        avg_recall = sum(metrics[name]["Recall@10"] for name, _ in DATASETS) / len(DATASETS)
        avg = (avg_ndcg + avg_recall) / 2.0
        rows.append((step, metrics, avg_ndcg, avg_recall, avg))

    lines = [
        "# Nonhomo Full Parallel Sweep Summary at @10",
        "",
        f"Run root: `{run_root}`",
        f"Output dir: `{output_dir}`",
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
    (output_dir / "summary_at10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(run_root: Path, output_dir: Path, model_prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    if experiment_complete(output_dir, model_prefix):
        write_summary_at10(run_root, output_dir, model_prefix)
        print(f"Skipping completed experiment: {model_prefix}")
        return

    free_gpus = wait_for_free_gpus()
    steps_to_run = [step for step in TARGET_STEPS if not checkpoint_complete(output_dir, model_prefix, step)]
    selected_gpus = free_gpus[: len(steps_to_run)]
    processes = []
    for gpu_id, step in zip(selected_gpus, steps_to_run):
        cmd = build_eval_command(run_root, output_dir, model_prefix, step)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        proc = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc", "-lc", cmd],
            env=env,
            cwd="/storage/BioMedNLP/llm2vec",
        )
        processes.append((step, gpu_id, proc))
        print(f"Started {model_prefix} cp{step} on GPU {gpu_id}")

    failed = []
    for step, gpu_id, proc in processes:
        return_code = proc.wait()
        print(f"Finished {model_prefix} cp{step} on GPU {gpu_id} with code {return_code}")
        if return_code != 0:
            failed.append((step, gpu_id, return_code))

    write_summary_at10(run_root, output_dir, model_prefix)
    if failed:
        raise RuntimeError(f"Some checkpoint jobs failed for {model_prefix}: {failed}")


def main() -> None:
    for config in EXPERIMENTS:
        run_experiment(config["run_root"], config["output_dir"], config["model_prefix"])


if __name__ == "__main__":
    main()
