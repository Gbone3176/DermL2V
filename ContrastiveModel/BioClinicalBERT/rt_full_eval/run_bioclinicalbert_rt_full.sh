#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/BioMedNLP/llm2vec"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

/opt/conda/envs/l2v/bin/python \
  ContrastiveModel/PubMedBERT/rt_full_eval/eval_pubmedbert_rt_full.py \
  --run_dir "ContrastiveModel/BioClinicalBERT/output/20260430_102154_bioclinicalbert_pool-cls_b64_ga2_ep2.0_lr2e-05_scale20_rawtext" \
  --output_root "ContrastiveModel/BioClinicalBERT/rt_full_eval/output" \
  --batch_size 128 \
  --max_length 512 \
  --pooling "cls" \
  --no_query_instruction \
  --no_doc_separator \
  --summary_title "BioClinicalBERT RT Nonhomo-Full Summary"
