#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

RUN_DIR="${RUN_DIR:-ContrastiveModel/Qwen3Embedding8B/output}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/downstream/RT_text/nonhomo-full/Qwen3Embedding8B}"
MAX_SAMPLES_ARGS=()
if [[ "${RT_MAX_SAMPLES:-0}" != "0" ]]; then
  MAX_SAMPLES_ARGS=(--max_samples "${RT_MAX_SAMPLES}")
fi

/opt/conda/envs/qwen3/bin/python \
  ContrastiveModel/Qwen3Embedding8B/rt_full_eval/eval_qwen3embedding8b_rt_full.py \
  --run_dir "${RUN_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --batch_size "${RT_BATCH_SIZE:-4}" \
  --max_length "${RT_MAX_LENGTH:-512}" \
  "${MAX_SAMPLES_ARGS[@]}"
