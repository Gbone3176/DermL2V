#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

/opt/conda/envs/l2v/bin/python \
  ContrastiveModel/PubMedBERT/rt_full_eval/eval_pubmedbert_rt_full.py \
  --run_dir "ContrastiveModel/PubMedBERT/output/20260429_143552_pubmedbert-base-embeddings_pool-mean_b64_ga4_ep2.0_lr2e-05_scale20" \
  --output_root "ContrastiveModel/PubMedBERT/rt_full_eval/output" \
  --batch_size 128 \
  --max_length 512
