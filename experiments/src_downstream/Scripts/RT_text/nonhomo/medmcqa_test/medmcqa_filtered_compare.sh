#!/usr/bin/env bash

CUDA_DEVICE=0
LLM2VEC_PYTHON="/opt/conda/envs/l2v/bin/python"
QWEN_PYTHON="/opt/conda/envs/qwen3/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/RT_text/medmcqa_filtered"
MAX_LENGTH=512
BATCH_SIZE=64

DATASET_DIR="/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text"
DATASET_FILES=(
    "${DATASET_DIR}/MedMCQA_RT_long_doc_lt_150.jsonl"
    "${DATASET_DIR}/MedMCQA_RT_long_doc_lt_200.jsonl"
    "${DATASET_DIR}/MedMCQA_RT_long_doc_lt_250.jsonl"
    "${DATASET_DIR}/MedMCQA_RT_mix_long_lt_150_short.jsonl"
    "${DATASET_DIR}/MedMCQA_RT_mix_long_lt_200_short.jsonl"
    "${DATASET_DIR}/MedMCQA_RT_mix_long_lt_250_short.jsonl"
)

QWEN_MODEL_NAME="Qwen3-Embedding-8B"
QWEN_MODEL_PATH="/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B"
ATTN_IMPLEMENTATION="sdpa"

LLM2VEC_MODEL_NAME="LLM2Vec_Llama-31-8B-ShareTopKSlerpMixCSE-noinst"
LLM2VEC_ADAPTER_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_TopKSharedSlerpMixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-100"
INSTRUCTION=""
POOLING_MODE="mean"

for DATASET_FILE in "${DATASET_FILES[@]}"; do
    echo "Running ${QWEN_MODEL_NAME} on ${DATASET_FILE}"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${QWEN_PYTHON}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_qwen_full \
        --input "${DATASET_FILE}" \
        --model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
        --model_name "${QWEN_MODEL_NAME}" \
        --batch_size ${BATCH_SIZE} \
        --max_length ${MAX_LENGTH} \
        --attn_implementation "${ATTN_IMPLEMENTATION}" \
        --output "${OUTPUT_DIR}"

    echo "Running ${LLM2VEC_MODEL_NAME} on ${DATASET_FILE}"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${LLM2VEC_PYTHON}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_full \
        --instruction "${INSTRUCTION}" \
        --dataset_file_path "${DATASET_FILE}" \
        --model_name "${LLM2VEC_MODEL_NAME}" \
        --pooling_mode "${POOLING_MODE}" \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BATCH_SIZE} \
        --enable_bidirectional True \
        --base_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
        --peft_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
        --extra_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" \
        --output "${OUTPUT_DIR}"
done
