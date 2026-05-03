#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-ContrastiveModel/Qwen3Embedding8B/output/20260501_061253_qwen3embedding8b_lora-r16_a32_b8_ga16_ep1.0_lr1e-05_tau0.01_fp16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ContrastiveModel/Qwen3Embedding8B/rt_full_eval/output}"
BATCH_SIZE="${RT_BATCH_SIZE:-4}"
MAX_LENGTH="${RT_MAX_LENGTH:-512}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/qwen3/bin/python}"
EVAL_SCRIPT="ContrastiveModel/Qwen3Embedding8B/rt_full_eval/eval_qwen3embedding8b_rt_full.py"

run_eval_group() {
  local gpu="$1"
  local summary_name="$2"
  shift 2
  local checkpoint_args=()
  for checkpoint in "$@"; do
    checkpoint_args+=(--checkpoint "${checkpoint}")
  done
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --run_dir "${RUN_DIR}" \
    --output_root "${OUTPUT_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --summary_name "${summary_name}" \
    "${checkpoint_args[@]}"
}

echo "[qwen3_rtfull_25_175] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_eval_group 3 summary_gpu3.md checkpoint-25 checkpoint-125 &
pid1=$!
run_eval_group 4 summary_gpu4.md checkpoint-50 checkpoint-150 &
pid2=$!
run_eval_group 5 summary_gpu5.md checkpoint-75 checkpoint-175 &
pid3=$!
run_eval_group 6 summary_gpu6.md checkpoint-100 &
pid4=$!

wait "${pid1}" "${pid2}" "${pid3}" "${pid4}"
echo "[qwen3_rtfull_25_175] worker groups complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"

run_eval_group 3 summary_at10.md \
  checkpoint-25 checkpoint-50 checkpoint-75 checkpoint-100 checkpoint-125 checkpoint-150 checkpoint-175
echo "[qwen3_rtfull_25_175] summary complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
