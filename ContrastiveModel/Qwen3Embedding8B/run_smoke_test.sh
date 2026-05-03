#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export NPROC_PER_NODE=1
export OUTPUT_ROOT="${OUTPUT_ROOT:-ContrastiveModel/Qwen3Embedding8B/debug_runs}"
export MAX_LENGTH="${MAX_LENGTH:-128}"
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export NUM_TRAIN_EPOCHS=1
export SAVE_STEPS=0
export EVAL_BATCH_SIZE=1
export RT_BATCH_SIZE=1
export RT_MAX_LENGTH="${RT_MAX_LENGTH:-128}"
export RT_MAX_SAMPLES="${RT_MAX_SAMPLES:-10}"

/opt/conda/envs/qwen3/bin/torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT:-29511}" \
  ContrastiveModel/Qwen3Embedding8B/train_qwen3embedding8b_lora.py \
  --model_name_or_path "/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B" \
  --data_dir "/storage/dataset/dermatoscop/Derm1M/DermVariantsData" \
  --output_root "${OUTPUT_ROOT}" \
  --max_length "${MAX_LENGTH}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps 2 \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --temperature "${TEMPERATURE:-0.01}" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
  --save_steps "${SAVE_STEPS}" \
  --local_files_only \
  --disable_swanlab \
  --eval_split "validation" \
  --eval_every_steps 0 \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --eval_max_samples 4 \
  --gradient_checkpointing \
  --fp16

RUN_DIR="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
unset OUTPUT_ROOT
RUN_DIR="${RUN_DIR}" OUTPUT_ROOT="${RT_OUTPUT_ROOT:-ContrastiveModel/Qwen3Embedding8B/rt_full_eval/debug_output}" \
  ContrastiveModel/Qwen3Embedding8B/run_rt_full.sh
