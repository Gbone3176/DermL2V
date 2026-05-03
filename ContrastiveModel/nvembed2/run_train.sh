#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ContrastiveModel/nvembed2/output}"

/opt/conda/envs/l2v/bin/torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT:-29512}" \
  ContrastiveModel/nvembed2/train_nvembed2_lora.py \
  --model_name_or_path "/cache/modelscope/models/nv-community/NV-Embed-v2" \
  --data_dir "/storage/dataset/dermatoscop/Derm1M/DermVariantsData" \
  --output_root "${OUTPUT_ROOT}" \
  --max_length "${MAX_LENGTH:-512}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-32}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.03}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --temperature "${TEMPERATURE:-0.01}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.1}" \
  --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}" \
  --save_steps "${SAVE_STEPS:-25}" \
  --local_files_only \
  --swanlab_project "Contrastive Model fine-tune" \
  --eval_split "validation" \
  --eval_every_steps "${EVAL_EVERY_STEPS:-0}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
  --gradient_checkpointing \
  --fp16
