#!/usr/bin/env bash

CUDA_DEVICE=4
PYTHON_BIN="/opt/conda/envs/qwen3/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full"
MAX_LENGTH=512
BATCH_SIZE=4
ATTN_IMPLEMENTATION="sdpa"

MODEL_NAME="NV-Embed-v2"
MODEL_PATH="nvidia/NV-Embed-v2"

DATASET_FILES=(
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl"
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedMCQA_RT_query_doc.jsonl"
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedQuAD_dermatology_qa_retrieval.jsonl"
)

for DATASET_FILE in "${DATASET_FILES[@]}"; do
    echo "Running NV-Embed-v2 full retrieval on ${DATASET_FILE}"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_nvembed_full \
        --input "${DATASET_FILE}" \
        --model_name_or_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --batch_size ${BATCH_SIZE} \
        --max_length ${MAX_LENGTH} \
        --attn_implementation "${ATTN_IMPLEMENTATION}" \
        --output "${OUTPUT_DIR}"
done
