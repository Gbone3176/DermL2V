INSTRUCTION="${INSTRUCTION:-Given a dermatologic question, return the answer that most closely corresponds to the information being asked for.}"
# INSTRUCTION="Given a question related to dermatology, retrieve the most relevant answer."
USE_INST="${USE_INST:-1}"

DOC_ADD_INST="${DOC_ADD_INST:-0}"

DEVICE_NUM="${DEVICE_NUM:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/bowenguo/.conda/envs/l2v/bin/python}"
RT_MODULE="${RT_MODULE:-experiments.src_downstream.rt_text.nonhomo.nonhomo_RT_l2v}"
DATASET_FILE="${DATASET_FILE:-/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/eval3-text-benchmark_split_choices.jsonl}"

DEVICE_COUNT=$(echo "$DEVICE_NUM" | awk -F',' '{print NF}')
if [ -z "${BATCH_SIZE:-}" ]; then
    if [ "$DEVICE_COUNT" -eq 2 ]; then
        BATCH_SIZE=96
    else
        BATCH_SIZE=64
    fi
fi
echo "BATCH_SIZE: $BATCH_SIZE"
echo "RT_MODULE: $RT_MODULE"



# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2304_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_DataV2"
# CPS=()
# for ((i=10; i<=290; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_ResCrossAttn_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_ResCrossAttn_DataV2"
# CPS=()
# for ((i=10; i<=330; i+=10)); do
#     CPS+=($i)
# done


# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_ResCrossAttn_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_DataV2"
# CPS=()
# for ((i=10; i<=445; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_Focal_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-5_s-42_w-10_lr-1e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_Focal_DataV2"
# CPS=()
# for ((i=10; i<=530; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_FocalMixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_FocalMixCSE_DataV2"
# CPS=()
# for ((i=10; i<=330; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_FocalMixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
# MODEL_NAME="withEval_QAx10_SlerpMixCSE_DermData2"
# CPS=()
# for ((i=10; i<=310; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_TopKSharedSlerpMixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
# MODEL_NAME="withEval_QAx10_ShareTopKSlerpMixCSE_DermData2"
# CPS=()
# for ((i=10; i<=310; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_TopKSharedSlerpMixCSE_DermData2_inst-query/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
# MODEL_NAME="withEval_QAx10_ShareTopKSlerpMixCSE_DermData2_inst-query"
# CPS=()
# for ((i=10; i<=130; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/StructuredSelfAttn_QAx10_SlerpMixCSE_query-inst/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16"
# CPS=()
# for ((i=10; i<=130; i+=10)); do
#     CPS+=($i)
# done

DERMA_MODEL_PATH="/mnt/nas1/disk06/bowenguo/codes/DermL2V/output/Llama31_8b_mntp-supervised/DermVariants/StructuredSelfAttn_QAx10_SlerpMixCSE_query-inst_uni-init/DermVariants_train_m-Meta-Llama-31-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16"
CPS=()
for ((i=10; i<=198; i+=10)); do
    CPS+=($i)
done


# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_ResCrossAttn_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-2048_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_ResCrossAttn_DataV2"
# CPS=()
# for ((i=10; i<=330; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-3_s-42_w-0_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_DataV1"
# CPS=()
# for ((i=5; i<=195; i+=5)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-5_s-42_w-0_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_DataV1"
# CPS=()
# for ((i=5; i<=225; i+=5)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_ResCrossAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-768_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_ResCrossAttn_DataV1"
# for ((i=200; i<=320; i+=20)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_ResCrossAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-768_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_ResCrossAttn_DataV1"
# for ((i=200; i<=320; i+=20)); do
#     CPS+=($i)
# done

MODEL_NAME="$(basename "$(dirname "$DERMA_MODEL_PATH")")"
if [ "$DOC_ADD_INST" -eq 1 ]; then
    MODEL_NAME="${MODEL_NAME}_DOC_ADD_INST"
fi
echo "MODEL_NAME: $MODEL_NAME"

if [ "$USE_INST" -eq 1 ]; then
    EVAL_INSTRUCTION="$INSTRUCTION"
    OUT_MODE="inst"
else
    EVAL_INSTRUCTION=""
    OUT_MODE="woinst"
fi
DOC_ADD_INST_FLAG=$([ "$DOC_ADD_INST" -eq 1 ] && echo "True" || echo "False")

POOLING_MODE=$(echo "$DERMA_MODEL_PATH" | sed -n 's/.*_p-\(.*\)_b-.*/\1/p')
echo "POOLING_MODE: $POOLING_MODE"

OUT_ROOT_BASE="${OUT_ROOT_BASE:-output/downstream/RT_text}"
OUT_ROOT="${OUT_ROOT:-${OUT_ROOT_BASE}/${MODEL_NAME}/${OUT_MODE}/}"
mkdir -p "$OUT_ROOT"

BASE_MODEL_NAME_OR_PATH="/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct"
PEFT_MODEL_NAME_OR_PATH="/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
EXTRA_MODEL_NAME_OR_PATH="/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"
SELFATTN_ATTN_HIDDEN_DIM=512
SELFATTN_NUM_HOPS=8
SELFATTN_OUTPUT_DROPOUT=0.0
SELFATTN_OUTPUT_LAYERNORM=True

# ck=0 直接表示使用基础模型进行评估, 固定pooling方式为mean
CUDA_VISIBLE_DEVICES=${DEVICE_NUM} "${PYTHON_BIN}" -m "$RT_MODULE" \
    --dataset_file_path "$DATASET_FILE" \
    --model_name "${MODEL_NAME}_cp_0" \
    --instruction "$EVAL_INSTRUCTION" \
    --doc_add_instruction "$DOC_ADD_INST_FLAG" \
    --pooling_mode "mean" \
    --max_length 512 \
    --batch_size "$BATCH_SIZE" \
    --enable_bidirectional True \
    --base_model_name_or_path $BASE_MODEL_NAME_OR_PATH \
    --peft_model_name_or_path $PEFT_MODEL_NAME_OR_PATH \
    --extra_model_name_or_path $EXTRA_MODEL_NAME_OR_PATH \
    --output "$OUT_ROOT"


for CP in "${CPS[@]}"; do
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} "${PYTHON_BIN}" -m "$RT_MODULE" \
        --dataset_file_path "$DATASET_FILE" \
        --model_name "${MODEL_NAME}_cp_${CP}" \
        --instruction "$EVAL_INSTRUCTION" \
        --doc_add_instruction "$DOC_ADD_INST_FLAG" \
        --pooling_mode "$POOLING_MODE" \
        --max_length 512 \
        --batch_size "$BATCH_SIZE" \
        --enable_bidirectional True \
        --selfattn_attn_hidden_dim "$SELFATTN_ATTN_HIDDEN_DIM" \
        --selfattn_num_hops "$SELFATTN_NUM_HOPS" \
        --selfattn_output_dropout "$SELFATTN_OUTPUT_DROPOUT" \
        --selfattn_output_layernorm "$SELFATTN_OUTPUT_LAYERNORM" \
        --base_model_name_or_path $BASE_MODEL_NAME_OR_PATH \
        --peft_model_name_or_path $PEFT_MODEL_NAME_OR_PATH \
        --extra_model_name_or_path $EXTRA_MODEL_NAME_OR_PATH "${DERMA_MODEL_PATH}/checkpoint-${CP}"\
        --output "$OUT_ROOT"
done
