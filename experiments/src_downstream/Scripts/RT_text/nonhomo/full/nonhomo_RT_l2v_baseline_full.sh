#!/usr/bin/env bash

# Full non-homogeneous RT retrieval test script for baseline DermL2V.
# Results will be saved as: ${OUTPUT_DIR}/<dataset_name>/<MODEL_NAME>.json
# Path values in this script should follow local_info/local_path.md.

CUDA_DEVICE=4
PYTHON_BIN="/opt/conda/envs/l2v/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/DermL2V/RT_text/nonhomo_full"
RT_DATA_ROOT="/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text"

INSTRUCTION="Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
POOLING_MODE="mean"
MAX_LENGTH=512
BATCH_SIZE=64

DATASET_FILES=(
    "${RT_DATA_ROOT}/eval3-text-benchmark_split_choices.jsonl"
    "${RT_DATA_ROOT}/MedMCQA_RT_query_doc.jsonl"
    "${RT_DATA_ROOT}/MedQuAD_dermatology_qa_retrieval.jsonl"
)

BASE_MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
SUPERVISED_MODEL_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"
DERML2V_ADAPTER_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/baseline/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-130"
MODEL_NAME="DermL2V_Baseline_cp130"

for DATASET_FILE in "${DATASET_FILES[@]}"; do
    echo "Running baseline DermL2V full retrieval on ${DATASET_FILE}"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_full \
        --instruction "${INSTRUCTION}" \
        --dataset_file_path "${DATASET_FILE}" \
        --model_name "${MODEL_NAME}" \
        --pooling_mode "${POOLING_MODE}" \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BATCH_SIZE} \
        --enable_bidirectional True \
        --base_model_name_or_path "${BASE_MODEL_PATH}" \
        --peft_model_name_or_path "${PEFT_MODEL_PATH}" \
        --extra_model_name_or_path "${SUPERVISED_MODEL_PATH}" "${DERML2V_ADAPTER_PATH}" \
        --output "${OUTPUT_DIR}"
done
