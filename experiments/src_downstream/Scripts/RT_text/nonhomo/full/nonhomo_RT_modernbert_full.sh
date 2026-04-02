#!/usr/bin/env bash

CUDA_DEVICE=0
PYTHON_BIN="/opt/conda/envs/qwen3/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full"
MAX_LENGTH=512
BATCH_SIZE=64
RT_DATA_ROOT="/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text"

dataset_dir() {
    case "$(basename "$1" .jsonl)" in
        eval3-text-benchmark_split_choices) echo "DermSynth_knowledgebase" ;;
        MedMCQA_RT_query_doc) echo "MedMCQA_RT" ;;
        MedQuAD_dermatology_qa_retrieval_doclt300) echo "MedQuAD_dermatology_qa_retrieval_doclt300" ;;
        *) basename "$1" .jsonl ;;
    esac
}

MODEL_SPECS=(
    "Clinical_ModernBERT|Simonlee711/Clinical_ModernBERT"
    "BioClinical-ModernBERT-large|OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1"
)

DATASET_FILES=(
    "${RT_DATA_ROOT}/eval3-text-benchmark_split_choices.jsonl"
    "${RT_DATA_ROOT}/MedMCQA_RT_query_doc.jsonl"
    "${RT_DATA_ROOT}/MedQuAD_dermatology_qa_retrieval_doclt300.jsonl"
)

for MODEL_SPEC in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r MODEL_NAME MODEL_PATH <<< "${MODEL_SPEC}"
    for DATASET_FILE in "${DATASET_FILES[@]}"; do
        echo "Running ${MODEL_NAME} full retrieval on ${DATASET_FILE}"
        rm -f "${OUTPUT_DIR}/$(dataset_dir "${DATASET_FILE}")/${MODEL_NAME}.json"
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_modernbert_full \
            --input "${DATASET_FILE}" \
            --model_path "${MODEL_PATH}" \
            --model_name "${MODEL_NAME}" \
            --max_length ${MAX_LENGTH} \
            --batch_size ${BATCH_SIZE} \
            --output "${OUTPUT_DIR}"
    done
done
