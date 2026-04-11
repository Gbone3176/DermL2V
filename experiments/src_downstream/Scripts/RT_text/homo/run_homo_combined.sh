#!/usr/bin/env bash
set -euo pipefail

ROOT="/storage/BioMedNLP/llm2vec"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/downstream/RT_text/homo/combined}"
VIS_DATASET="${VIS_DATASET:-/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl}"
DERMVARIANTS_DIR="${DERMVARIANTS_DIR:-/storage/dataset/dermatoscop/Derm1M/DermVariantsData}"

L2V_PYTHON="${L2V_PYTHON:-/opt/conda/envs/l2v/bin/python}"
QWEN_PYTHON="${QWEN_PYTHON:-/opt/conda/envs/qwen3/bin/python}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-separate}"

mkdir -p "$OUTPUT_DIR"

# Baselines
CUDA_VISIBLE_DEVICES="${GPU_BERT:-0}" "$L2V_PYTHON" -m experiments.src_downstream.rt_text.homo.homo_RT_bert \
  --model_path "emilyalsentzer/Bio_ClinicalBERT" \
  --model_name "BioClinicalBERT" \
  --vis_dataset "$VIS_DATASET" \
  --dermvariants_dir "$DERMVARIANTS_DIR" \
  --retrieval_mode "$RETRIEVAL_MODE" \
  --output "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${GPU_BIOLINKBERT:-1}" "$L2V_PYTHON" -m experiments.src_downstream.rt_text.homo.homo_RT_bert \
  --model_path "michiyasunaga/BioLinkBERT-large" \
  --model_name "BioLinkBERT" \
  --vis_dataset "$VIS_DATASET" \
  --dermvariants_dir "$DERMVARIANTS_DIR" \
  --retrieval_mode "$RETRIEVAL_MODE" \
  --output "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${GPU_PUBMEDBERT:-2}" "$L2V_PYTHON" -m experiments.src_downstream.rt_text.homo.homo_RT_bert \
  --model_path "NeuML/pubmedbert-base-embeddings" \
  --model_name "pubmedbert-base-embeddings" \
  --vis_dataset "$VIS_DATASET" \
  --dermvariants_dir "$DERMVARIANTS_DIR" \
  --retrieval_mode "$RETRIEVAL_MODE" \
  --output "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${GPU_GPT2:-3}" "$L2V_PYTHON" -m experiments.src_downstream.rt_text.homo.homo_RT_gpt2 \
  --model_path "openai-community/gpt2" \
  --model_name "gpt2" \
  --vis_dataset "$VIS_DATASET" \
  --dermvariants_dir "$DERMVARIANTS_DIR" \
  --retrieval_mode "$RETRIEVAL_MODE" \
  --output "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${GPU_MODERNBERT:-4}" "$QWEN_PYTHON" -m experiments.src_downstream.rt_text.homo.homo_RT_modernbert \
  --model_path "Simonlee711/Clinical_ModernBERT" \
  --model_name "Clinical_ModernBERT" \
  --vis_dataset "$VIS_DATASET" \
  --dermvariants_dir "$DERMVARIANTS_DIR" \
  --retrieval_mode "$RETRIEVAL_MODE" \
  --output "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="${GPU_MODERNBERT_LARGE:-5}" "$QWEN_PYTHON" -m experiments.src_downstream.rt_text.homo.homo_RT_modernbert \
  --model_path "OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1" \
  --model_name "BioClinical-ModernBERT-large" \
  --vis_dataset "$VIS_DATASET" \
  --dermvariants_dir "$DERMVARIANTS_DIR" \
  --retrieval_mode "$RETRIEVAL_MODE" \
  --output "$OUTPUT_DIR"

# DermL2V example
# Uncomment and adjust checkpoint path if you want to run the current best homo combined setting.
# CUDA_VISIBLE_DEVICES="${GPU_L2V:-6}" "$L2V_PYTHON" -m experiments.src_downstream.rt_text.homo.homo_RT_l2v \
#   --model_name "withEval_QAx10_TopKSharedSlerpMixCSE_DermData2_inst-query_cp-130" \
#   --pooling_mode "mean" \
#   --batch_size 8 \
#   --vis_dataset "$VIS_DATASET" \
#   --dermvariants_dir "$DERMVARIANTS_DIR" \
#   --retrieval_mode "$RETRIEVAL_MODE" \
#   --base_model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
#   --extra_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_TopKSharedSlerpMixCSE_DermData2_inst-query/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-130" \
#   --output "$OUTPUT_DIR"
