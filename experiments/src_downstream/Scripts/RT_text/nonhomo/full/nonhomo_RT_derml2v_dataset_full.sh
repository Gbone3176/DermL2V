#!/usr/bin/env bash

set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="/opt/conda/envs/l2v/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full"

INSTRUCTION="Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
BASE_MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
SUPERVISED_MODEL_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"

MODEL_NAME="${1:?model name required}"
POOLING_MODE="${2:?pooling mode required}"
ADAPTER_PATH="${3:?adapter path required}"
DATASET_FILE="${4:?dataset file required}"

MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"

dataset_dir() {
    case "$(basename "$1" .jsonl)" in
        eval3-text-benchmark_split_choices) echo "DermSynth_knowledgebase" ;;
        MedMCQA_RT_query_doc) echo "MedMCQA_RT" ;;
        MedQuAD_dermatology_qa_retrieval_doclt300) echo "MedQuAD_dermatology_qa_retrieval_doclt300" ;;
        *) basename "$1" .jsonl ;;
    esac
}

output_file="${OUTPUT_DIR}/$(dataset_dir "${DATASET_FILE}")/${MODEL_NAME}.json"
rm -f "${output_file}"
echo "Running ${MODEL_NAME} on ${DATASET_FILE}"

if [[ "${POOLING_MODE}" == "structured_selfattn" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_full \
        --instruction "${INSTRUCTION}" \
        --dataset_file_path "${DATASET_FILE}" \
        --model_name "${MODEL_NAME}" \
        --pooling_mode "${POOLING_MODE}" \
        --max_length "${MAX_LENGTH}" \
        --batch_size "${BATCH_SIZE}" \
        --enable_bidirectional True \
        --selfattn_attn_hidden_dim 512 \
        --selfattn_num_hops 8 \
        --selfattn_output_dropout 0.0 \
        --selfattn_output_layernorm True \
        --base_model_name_or_path "${BASE_MODEL_PATH}" \
        --peft_model_name_or_path "${PEFT_MODEL_PATH}" \
        --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "${ADAPTER_PATH}" \
        --output "${OUTPUT_DIR}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_full \
        --instruction "${INSTRUCTION}" \
        --dataset_file_path "${DATASET_FILE}" \
        --model_name "${MODEL_NAME}" \
        --pooling_mode "${POOLING_MODE}" \
        --max_length "${MAX_LENGTH}" \
        --batch_size "${BATCH_SIZE}" \
        --enable_bidirectional True \
        --base_model_name_or_path "${BASE_MODEL_PATH}" \
        --peft_model_name_or_path "${PEFT_MODEL_PATH}" \
        --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "${ADAPTER_PATH}" \
        --output "${OUTPUT_DIR}"
fi
