#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ContrastiveModel/Qwen3Embedding8B/output}"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

/opt/conda/envs/qwen3/bin/torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT:-29511}" \
  ContrastiveModel/Qwen3Embedding8B/train_qwen3embedding8b_lora.py \
  --model_name_or_path "/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B" \
  --data_dir "/storage/dataset/dermatoscop/Derm1M/DermVariantsData" \
  --output_root "${OUTPUT_ROOT}" \
  --max_length "${MAX_LENGTH:-512}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --temperature "${TEMPERATURE:-0.01}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}" \
  --save_steps "${SAVE_STEPS:-25}" \
  --local_files_only \
  --swanlab_project "Contrastive Model fine-tune" \
  --eval_split "validation" \
  --eval_every_steps "${EVAL_EVERY_STEPS:-0}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
  "${RESUME_ARGS[@]}" \
  --gradient_checkpointing \
  --fp16

if [[ "${RUN_RT_AFTER_TRAIN:-1}" == "1" ]]; then
  RUN_DIR="$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  unset OUTPUT_ROOT
  RUN_DIR="${RUN_DIR}" OUTPUT_ROOT="${RT_OUTPUT_ROOT:-output/downstream/RT_text/nonhomo-full/Qwen3Embedding8B}" \
    ContrastiveModel/Qwen3Embedding8B/run_rt_full.sh
fi
