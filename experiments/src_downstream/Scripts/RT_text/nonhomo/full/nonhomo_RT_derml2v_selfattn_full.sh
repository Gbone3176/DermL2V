#!/usr/bin/env bash

set -euo pipefail

CUDA_DEVICE="${CUDA_DEVICE:-6}"
PYTHON_BIN="/opt/conda/envs/l2v/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full"
RT_DATA_ROOT="/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text"

INSTRUCTION="Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
POOLING_MODE="structured_selfattn"
MAX_LENGTH=512
BATCH_SIZE=64
SELFATTN_ATTN_HIDDEN_DIM=512
SELFATTN_NUM_HOPS=8
SELFATTN_OUTPUT_DROPOUT=0.0
SELFATTN_OUTPUT_NORM=layernorm

DATASET_FILES=(
    "${RT_DATA_ROOT}/eval3-text-benchmark_split_choices.jsonl"
    "${RT_DATA_ROOT}/MedMCQA_RT_query_doc.jsonl"
    "${RT_DATA_ROOT}/MedQuAD_dermatology_qa_retrieval_doclt300.jsonl"
)

dataset_dir() {
    case "$(basename "$1" .jsonl)" in
        eval3-text-benchmark_split_choices) echo "DermSynth_knowledgebase" ;;
        MedMCQA_RT_query_doc) echo "MedMCQA_RT" ;;
        MedQuAD_dermatology_qa_retrieval_doclt300) echo "MedQuAD_dermatology_qa_retrieval_doclt300" ;;
        *) basename "$1" .jsonl ;;
    esac
}

BASE_MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
SUPERVISED_MODEL_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"

run_model() {
    local model_name="$1"
    local adapter_path="$2"
    for dataset_file in "${DATASET_FILES[@]}"; do
        echo "Running ${model_name} on ${dataset_file}"
        rm -f "${OUTPUT_DIR}/$(dataset_dir "${dataset_file}")/${model_name}.json"
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_full \
            --instruction "${INSTRUCTION}" \
            --dataset_file_path "${dataset_file}" \
            --model_name "${model_name}" \
            --pooling_mode "${POOLING_MODE}" \
            --max_length "${MAX_LENGTH}" \
            --batch_size "${BATCH_SIZE}" \
            --enable_bidirectional True \
            --selfattn_attn_hidden_dim "${SELFATTN_ATTN_HIDDEN_DIM}" \
            --selfattn_num_hops "${SELFATTN_NUM_HOPS}" \
            --selfattn_output_dropout "${SELFATTN_OUTPUT_DROPOUT}" \
            --selfattn_output_norm "${SELFATTN_OUTPUT_NORM}" \
            --base_model_name_or_path "${BASE_MODEL_PATH}" \
            --peft_model_name_or_path "${PEFT_MODEL_PATH}" \
            --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "${adapter_path}" \
            --output "${OUTPUT_DIR}"
    done
}

if [[ -n "${MODEL_NAME_OVERRIDE:-}" && -n "${ADAPTER_PATH_OVERRIDE:-}" ]]; then
    run_model "${MODEL_NAME_OVERRIDE}" "${ADAPTER_PATH_OVERRIDE}"
    exit 0
fi

run_model "DermL2V_Baseline_SM_SA_K8_cp50" \
    "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k8_StructuredSelfAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"

run_model "DermL2V_Baseline_SM_SA_K16_cp50" \
    "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k16_StructuredSelfAttn/DermVariants_train_m-Meta-Llama-31-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"

run_model "DermL2V_Baseline_SM_SA_K32_cp50" \
    "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k32_StructuredSelfAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"

run_model "DermL2V_Baseline_SM_SA_K64_cp50" \
    "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k64_StructuredSelfAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
