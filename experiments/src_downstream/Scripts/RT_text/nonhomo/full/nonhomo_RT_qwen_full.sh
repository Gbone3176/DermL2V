#!/usr/bin/env bash

CUDA_DEVICE=1
PYTHON_BIN="/opt/conda/envs/qwen3/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/RT_text/nonhomo/full"
MAX_LENGTH=512
BATCH_SIZE=64
ATTN_IMPLEMENTATION="sdpa"

# MODEL_NAME="Qwen3-Embedding-0.6B"
# MODEL_PATH="/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B"
    
MODEL_NAME="Qwen3-Embedding-8B"
MODEL_PATH="/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B"

DATASET_FILES=(
    "/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/eval3-text-benchmark_split_choices.jsonl"
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedMCQA_RT_query_doc.jsonl"
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedQuAD_dermatology_qa_retrieval.jsonl"
)

for DATASET_FILE in "${DATASET_FILES[@]}"; do
    echo "Running Qwen full retrieval on ${DATASET_FILE}"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_qwen_full \
        --input "${DATASET_FILE}" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --batch_size ${BATCH_SIZE} \
        --max_length ${MAX_LENGTH} \
        --attn_implementation "${ATTN_IMPLEMENTATION}" \
        --output "${OUTPUT_DIR}"
done
