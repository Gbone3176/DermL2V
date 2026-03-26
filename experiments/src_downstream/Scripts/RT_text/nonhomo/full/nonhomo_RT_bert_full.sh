#!/usr/bin/env bash

CUDA_DEVICE=0
PYTHON_BIN="/opt/conda/envs/l2v/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/RT_text/full"
MAX_LENGTH=512
BATCH_SIZE=64

MODEL_SPECS=(
    "BioClinicalBERT|emilyalsentzer/Bio_ClinicalBERT"
    "BioLinkBERT|michiyasunaga/BioLinkBERT-large"
    "pubmedbert-base-embeddings|NeuML/pubmedbert-base-embeddings"
)

DATASET_FILES=(
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl"
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedMCQA_RT_query_doc.jsonl"
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedQuAD_dermatology_qa_retrieval.jsonl"
)

for MODEL_SPEC in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r MODEL_NAME MODEL_PATH <<< "${MODEL_SPEC}"
    for DATASET_FILE in "${DATASET_FILES[@]}"; do
        echo "Running ${MODEL_NAME} full retrieval on ${DATASET_FILE}"
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_bert_full \
            --input "${DATASET_FILE}" \
            --model_path "${MODEL_PATH}" \
            --model_name "${MODEL_NAME}" \
            --max_length ${MAX_LENGTH} \
            --batch_size ${BATCH_SIZE} \
            --output "${OUTPUT_DIR}"
    done
done
