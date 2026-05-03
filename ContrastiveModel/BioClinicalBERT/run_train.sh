#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

/opt/conda/envs/l2v/bin/torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  ContrastiveModel/BioClinicalBERT/train_bioclinicalbert.py \
  --model_name_or_path "emilyalsentzer/Bio_ClinicalBERT" \
  --data_dir "/storage/dataset/dermatoscop/Derm1M/DermVariantsData" \
  --output_root "ContrastiveModel/BioClinicalBERT/output" \
  --pooling "cls" \
  --max_length 512 \
  --num_train_epochs 2 \
  --per_device_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --loss_scale 20.0 \
  --save_steps 25 \
  --local_files_only \
  --swanlab_project "Contrastive Model fine-tune" \
  --eval_split "validation" \
  --eval_every_steps 0 \
  --eval_batch_size 128 \
  --gradient_checkpointing \
  --bf16
