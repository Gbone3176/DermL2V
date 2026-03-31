#!/usr/bin/env bash

set -euo pipefail

# Re-run all existing MedQuAD nonhomo_full experiments on the doc<300 variant.
# Path values in this script should follow local_info/local_path.md.

PYTHON_L2V="/opt/conda/envs/l2v/bin/python"
PYTHON_QWEN="/opt/conda/envs/qwen3/bin/python"

OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full"
DATASET_FILE="/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedQuAD_dermatology_qa_retrieval_doclt300.jsonl"
TARGET_DIR="${OUTPUT_DIR}/MedQuAD_dermatology_qa_retrieval_doclt300"

INSTRUCTION="Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
MAX_LENGTH=512
BATCH_SIZE_L2V=64
BATCH_SIZE_BERT=64
BATCH_SIZE_GPT2=32
BATCH_SIZE_QWEN=64
ATTN_IMPLEMENTATION="sdpa"

GPU_L2V=7
GPU_BERT=6
GPU_GPT2=6
GPU_MODERNBERT=6
GPU_QWEN=6

BASE_MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
SUPERVISED_MODEL_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"

mkdir -p "${TARGET_DIR}"

remove_result() {
    local model_name="$1"
    rm -f "${TARGET_DIR}/${model_name}.json"
}

run_l2v() {
    local model_name="$1"
    local pooling_mode="$2"
    shift 2

    echo "Running ${model_name}"
    remove_result "${model_name}"
    CUDA_VISIBLE_DEVICES="${GPU_L2V}" "${PYTHON_L2V}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_full \
        --instruction "${INSTRUCTION}" \
        --dataset_file_path "${DATASET_FILE}" \
        --model_name "${model_name}" \
        --pooling_mode "${pooling_mode}" \
        --max_length "${MAX_LENGTH}" \
        --batch_size "${BATCH_SIZE_L2V}" \
        --enable_bidirectional True \
        --base_model_name_or_path "${BASE_MODEL_PATH}" \
        --peft_model_name_or_path "${PEFT_MODEL_PATH}" \
        "$@" \
        --output "${OUTPUT_DIR}"
}

run_bert() {
    local model_name="$1"
    local model_path="$2"

    echo "Running ${model_name}"
    remove_result "${model_name}"
    CUDA_VISIBLE_DEVICES="${GPU_BERT}" "${PYTHON_L2V}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_bert_full \
        --input "${DATASET_FILE}" \
        --model_path "${model_path}" \
        --model_name "${model_name}" \
        --max_length "${MAX_LENGTH}" \
        --batch_size "${BATCH_SIZE_BERT}" \
        --output "${OUTPUT_DIR}"
}

run_gpt2() {
    echo "Running gpt2"
    remove_result "gpt2"
    CUDA_VISIBLE_DEVICES="${GPU_GPT2}" "${PYTHON_L2V}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_gpt2_full \
        --input "${DATASET_FILE}" \
        --model_path "openai-community/gpt2" \
        --model_name "gpt2" \
        --max_length "${MAX_LENGTH}" \
        --batch_size "${BATCH_SIZE_GPT2}" \
        --output "${OUTPUT_DIR}"
}

run_modernbert() {
    local model_name="$1"
    local model_path="$2"

    echo "Running ${model_name}"
    remove_result "${model_name}"
    CUDA_VISIBLE_DEVICES="${GPU_MODERNBERT}" "${PYTHON_QWEN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_modernbert_full \
        --input "${DATASET_FILE}" \
        --model_path "${model_path}" \
        --model_name "${model_name}" \
        --max_length "${MAX_LENGTH}" \
        --batch_size "${BATCH_SIZE_BERT}" \
        --output "${OUTPUT_DIR}"
}

run_qwen() {
    local model_name="$1"
    local model_path="$2"

    echo "Running ${model_name}"
    remove_result "${model_name}"
    CUDA_VISIBLE_DEVICES="${GPU_QWEN}" "${PYTHON_QWEN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_qwen_full \
        --input "${DATASET_FILE}" \
        --model_name_or_path "${model_path}" \
        --model_name "${model_name}" \
        --batch_size "${BATCH_SIZE_QWEN}" \
        --max_length "${MAX_LENGTH}" \
        --attn_implementation "${ATTN_IMPLEMENTATION}" \
        --output "${OUTPUT_DIR}"
}

run_bert "BioClinicalBERT" "emilyalsentzer/Bio_ClinicalBERT"
run_modernbert "BioClinical-ModernBERT-large" "OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1"
run_bert "BioLinkBERT" "michiyasunaga/BioLinkBERT-large"
run_modernbert "Clinical_ModernBERT" "Simonlee711/Clinical_ModernBERT"
run_l2v "DermL2V_Baseline_cp130" "mean" \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/baseline/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-130"
run_l2v "DermL2V_Baseline" "mean" \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/baseline/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
run_l2v "DermL2V_Baseline_SM_K16_cp130" "mean" \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k16/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-130"
run_l2v "DermL2V_Baseline_SM_K16_cp50" "mean" \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k16/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
run_l2v "DermL2V_Baseline_SM_SA_K16_cp50" "structured_selfattn" \
    --selfattn_attn_hidden_dim 512 \
    --selfattn_num_hops 8 \
    --selfattn_output_dropout 0.0 \
    --selfattn_output_layernorm True \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k16_StructuredSelfAttn/DermVariants_train_m-Meta-Llama-31-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
run_l2v "DermL2V_Baseline_SM_SA_K32_cp50" "structured_selfattn" \
    --selfattn_attn_hidden_dim 512 \
    --selfattn_num_hops 8 \
    --selfattn_output_dropout 0.0 \
    --selfattn_output_layernorm True \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k32_StructuredSelfAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
run_l2v "DermL2V_Baseline_SM_SA_K64_cp50" "structured_selfattn" \
    --selfattn_attn_hidden_dim 512 \
    --selfattn_num_hops 8 \
    --selfattn_output_dropout 0.0 \
    --selfattn_output_layernorm True \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k64_StructuredSelfAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
run_l2v "DermL2V_Baseline_SM_SA_K8_cp50" "structured_selfattn" \
    --selfattn_attn_hidden_dim 512 \
    --selfattn_num_hops 8 \
    --selfattn_output_dropout 0.0 \
    --selfattn_output_layernorm True \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SlerpMixCSE_k8_StructuredSelfAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50"
run_gpt2
run_l2v "LLM2Vec_Llama-31-8B" "mean" \
    --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}"
run_bert "pubmedbert-base-embeddings" "NeuML/pubmedbert-base-embeddings"
run_qwen "Qwen3-Embedding-0.6B" "/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B"
run_qwen "Qwen3-Embedding-8B" "/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B"

echo "All runs completed."
