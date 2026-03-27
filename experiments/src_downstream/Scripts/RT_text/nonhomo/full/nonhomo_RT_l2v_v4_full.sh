#!/usr/bin/env bash

# Full non-homogeneous RT retrieval test script for llm2vecV4.
# Results will be saved as: ${OUTPUT_DIR}/<dataset_name>/<MODEL_NAME>.json

CUDA_DEVICE=0
PYTHON_BIN="/opt/conda/envs/l2v/bin/python"
OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/RT_text/nonhomo/full"

INSTRUCTION="Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
POOLING_MODE="res_mlp_pooling"
MAX_LENGTH=512
BATCH_SIZE=64

# V4 residual MLP pooling options
RES_MLP_HIDDEN_DIM=1024
RES_MLP_NUM_LAYERS=4
RES_MLP_DROPOUT=0.0
RES_MLP_GAMMA_INIT=0.001
RES_MLP_GAMMA_LEARNABLE=True
RES_MLP_OUTPUT_NORMALIZE=True
RES_MLP_OUTPUT_LAYERNORM=True

DATASET_FILES=(
    "/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/eval3-text-benchmark_split_choices.jsonl"
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedMCQA_RT_query_doc.jsonl"
    "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedQuAD_dermatology_qa_retrieval.jsonl"
)

# Example V4 model path settings. Replace with your actual V4 checkpoint if needed.
BASE_MODEL_PATH="/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct"
PEFT_MODEL_PATH="/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
EXTRA_MODEL_PATHS=("/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db")
MODEL_NAME="withEval_QAx10_ShareTopKSlerpMixCSE_ResMLPPool_DermData2"

for DATASET_FILE in "${DATASET_FILES[@]}"; do
    echo "Running llm2vecV4 full retrieval on ${DATASET_FILE}"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.full.nonhomo_RT_l2v_v4_full \
        --instruction "${INSTRUCTION}" \
        --dataset_file_path "${DATASET_FILE}" \
        --model_name "${MODEL_NAME}" \
        --pooling_mode "${POOLING_MODE}" \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BATCH_SIZE} \
        --enable_bidirectional True \
        --res_mlp_hidden_dim ${RES_MLP_HIDDEN_DIM} \
        --res_mlp_num_layers ${RES_MLP_NUM_LAYERS} \
        --res_mlp_dropout ${RES_MLP_DROPOUT} \
        --res_mlp_gamma_init ${RES_MLP_GAMMA_INIT} \
        --res_mlp_gamma_learnable ${RES_MLP_GAMMA_LEARNABLE} \
        --res_mlp_output_normalize ${RES_MLP_OUTPUT_NORMALIZE} \
        --res_mlp_output_layernorm ${RES_MLP_OUTPUT_LAYERNORM} \
        --base_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
        --peft_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
        --extra_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" \
        --output "${OUTPUT_DIR}"
done
