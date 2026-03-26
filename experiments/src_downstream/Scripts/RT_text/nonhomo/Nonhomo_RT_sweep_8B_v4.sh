#!/usr/bin/env bash

INSTRUCTION="Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
USE_INST=1

DEVICE_NUM=0
PYTHON_BIN="/opt/conda/envs/l2v/bin/python"

DATASET_FILE="/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl"

DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/ResMLPPooling_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-res_mlp_pooling_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
MODEL_NAME="withEval_QAx10_ShareTopKSlerpMixCSE_ResMLPPool_DermData2"

CPS=()
for ((i=10; i<=140; i+=10)); do
    CPS+=($i)
done

POOLING_MODE="res_mlp_pooling"
OUT_ROOT="/storage/BioMedNLP/llm2vec/output/downstream/RT_text/${MODEL_NAME}/$([ "$USE_INST" -eq 1 ] && echo "inst" || echo "woinst")/"
mkdir -p "$OUT_ROOT"

BASE_MODEL_NAME_OR_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_NAME_OR_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
EXTRA_MODEL_BASE="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"

RES_MLP_HIDDEN_DIM=1024
RES_MLP_NUM_LAYERS=4
RES_MLP_DROPOUT=0.0
RES_MLP_GAMMA_INIT=0.001
RES_MLP_GAMMA_LEARNABLE=True
RES_MLP_OUTPUT_NORMALIZE=True
RES_MLP_OUTPUT_LAYERNORM=True

for CP in "${CPS[@]}"; do
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} "${PYTHON_BIN}" -m experiments.src_downstream.rt_text.nonhomo.nonhomo_RT_l2v_v4 \
        --dataset_file_path "${DATASET_FILE}" \
        --model_name "${MODEL_NAME}_cp_${CP}" \
        --instruction "$([ "$USE_INST" -eq 1 ] && echo "$INSTRUCTION" || echo "")" \
        --pooling_mode "${POOLING_MODE}" \
        --max_length 512 \
        --batch_size 64 \
        --enable_bidirectional True \
        --res_mlp_hidden_dim "${RES_MLP_HIDDEN_DIM}" \
        --res_mlp_num_layers "${RES_MLP_NUM_LAYERS}" \
        --res_mlp_dropout "${RES_MLP_DROPOUT}" \
        --res_mlp_gamma_init "${RES_MLP_GAMMA_INIT}" \
        --res_mlp_gamma_learnable "${RES_MLP_GAMMA_LEARNABLE}" \
        --res_mlp_output_normalize "${RES_MLP_OUTPUT_NORMALIZE}" \
        --res_mlp_output_layernorm "${RES_MLP_OUTPUT_LAYERNORM}" \
        --base_model_name_or_path "${BASE_MODEL_NAME_OR_PATH}" \
        --peft_model_name_or_path "${PEFT_MODEL_NAME_OR_PATH}" \
        --extra_model_name_or_path "${EXTRA_MODEL_BASE}" "${DERMA_MODEL_PATH}/checkpoint-${CP}" \
        --output "${OUT_ROOT}"
done
