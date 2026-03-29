#!/usr/bin/env bash

# Full non-homogeneous RT retrieval test script for structured self-attention pooling.
# Results will be saved as: ${OUTPUT_DIR}/<dataset_name>/<MODEL_NAME>.json

CUDA_DEVICE=0
OUTPUT_DIR="output/downstream/RT_text/full"

INSTRUCTION="Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
POOLING_MODE="structured_selfattn"
MAX_LENGTH=512
BATCH_SIZE=128

SELFATTN_ATTN_HIDDEN_DIM=512
SELFATTN_NUM_HOPS=8
SELFATTN_OUTPUT_DROPOUT=0.0
SELFATTN_OUTPUT_LAYERNORM=True

DATASET_FILES=(
    "/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/eval3-text-benchmark_split_choices.jsonl"
    "/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/MedMCQA_RT_query_doc.jsonl"
    "/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/MedQuAD_dermatology_qa_retrieval.jsonl"
)

BASE_MODEL_PATH="/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct"
PEFT_MODEL_PATH="/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
EXTRA_MODEL_PATHS=(
    "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"
    "/mnt/nas1/disk06/bowenguo/codes/DermL2V/output/Llama31_8b_mntp-supervised/DermVariants/StructuredSelfAttn_QAx10_SlerpMixCSE_query-inst_uni-init/DermVariants_train_m-Meta-Llama-31-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
)
MODEL_NAME="StructuredSelfAttn_QAx10_SlerpMixCSE_query-inst_uni-init"

for DATASET_FILE in "${DATASET_FILES[@]}"; do
    echo "Running llm2vec structured self-attention full retrieval on ${DATASET_FILE}"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_full \
        --instruction "${INSTRUCTION}" \
        --dataset_file_path "${DATASET_FILE}" \
        --model_name "${MODEL_NAME}" \
        --pooling_mode "${POOLING_MODE}" \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BATCH_SIZE} \
        --enable_bidirectional True \
        --selfattn_attn_hidden_dim ${SELFATTN_ATTN_HIDDEN_DIM} \
        --selfattn_num_hops ${SELFATTN_NUM_HOPS} \
        --selfattn_output_dropout ${SELFATTN_OUTPUT_DROPOUT} \
        --selfattn_output_layernorm ${SELFATTN_OUTPUT_LAYERNORM} \
        --base_model_name_or_path "${BASE_MODEL_PATH}" \
        --peft_model_name_or_path "${PEFT_MODEL_PATH}" \
        --extra_model_name_or_path "${EXTRA_MODEL_PATHS[@]}" \
        --output "${OUTPUT_DIR}"
done
