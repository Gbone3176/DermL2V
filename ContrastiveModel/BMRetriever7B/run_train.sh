#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

/opt/conda/envs/l2v/bin/torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  ContrastiveModel/BMRetriever7B/train_bmretriever7b_lora.py \
  --model_name_or_path "/cache/transformers_cache/models--BMRetriever--BMRetriever-7B/snapshots/13e6adb9273c5f254e037987d6b44e9e4b005b9a" \
  --data_dir "/storage/dataset/dermatoscop/Derm1M/DermVariantsData" \
  --output_root "ContrastiveModel/BMRetriever7B/output" \
  --max_length 512 \
  --num_train_epochs 1 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --temperature 1.0 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
  --save_steps 25 \
  --local_files_only \
  --swanlab_project "Contrastive Model fine-tune" \
  --eval_split "validation" \
  --eval_every_steps 0 \
  --eval_batch_size 4 \
  --gradient_checkpointing \
  --fp16
