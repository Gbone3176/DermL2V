#!/usr/bin/env bash

CUDA_DEVICE=1
PYTHON_BIN="/opt/conda/envs/qwen3/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full"
MAX_LENGTH=512
BATCH_SIZE=64
ATTN_IMPLEMENTATION="sdpa"
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
    "Qwen3-Embedding-0.6B|/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B"
    "Qwen3-Embedding-8B|/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B"
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
        CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_qwen_full \
            --input "${DATASET_FILE}" \
            --model_name_or_path "${MODEL_PATH}" \
            --model_name "${MODEL_NAME}" \
            --batch_size ${BATCH_SIZE} \
            --max_length ${MAX_LENGTH} \
            --attn_implementation "${ATTN_IMPLEMENTATION}" \
            --output "${OUTPUT_DIR}"
    done
done
