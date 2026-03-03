INSTRUCTION="Given a dermatologic question, return the answer that most closely corresponds to the information being asked for."
USE_INST=0

DEVICE_NUM=5

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-30_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_DataV2"
# CPS=()
# for ((i=10; i<=320; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_DataV2"
# CPS=()
# for ((i=10; i<=320; i+=10)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_ResCrossAttn_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-1024_l-512_bidirectional-True_e-5_s-42_w-250_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_ResCrossAttn_DataV2"
# CPS=()
# for ((i=20; i<=480; i+=20)); do
#     CPS+=($i)
# done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-3_s-42_w-0_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_DataV1"
# CPS=()
# for ((i=5; i<=195; i+=5)); do
#     CPS+=($i)
# done

DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-5_s-42_w-0_lr-2e-05_lora_r-16"
MODEL_NAME="Derml2v-8B_Baseline_MixCSE_DataV1"
CPS=()
for ((i=5; i<=225; i+=5)); do
    CPS+=($i)
done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_ResCrossAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-768_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_ResCrossAttn_DataV1"
# for ((i=200; i<=320; i+=20)); do
#     CPS+=($i)
# done

POOLING_MODE=$(echo "$DERMA_MODEL_PATH" | sed -n 's/.*_p-\(.*\)_b-.*/\1/p')
echo "POOLING_MODE: $POOLING_MODE"

OUT_ROOT="/storage/BioMedNLP/llm2vec/output/downstream/RT_text/${MODEL_NAME}/$([ "$USE_INST" -eq 1 ] && echo "inst" || echo "woinst")/"
mkdir -p "$OUT_ROOT"

BASE_MODEL_NAME_OR_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_NAME_OR_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
EXTRA_MODEL_NAME_OR_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"


CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python -m experiments.src_downstream.nonhomo_RT_l2v \
    --dataset_file_path "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl" \
    --model_name "${MODEL_NAME}_cp_0" \
    --instruction "$([ "$USE_INST" -eq 1 ] && echo "$INSTRUCTION" || echo "")" \
    --pooling_mode "$POOLING_MODE" \
    --max_length 512 \
    --batch_size 64 \
    --enable_bidirectional True \
    --base_model_name_or_path "$BASE_MODEL_NAME_OR_PATH" \
    --peft_model_name_or_path "$PEFT_MODEL_NAME_OR_PATH" \
    --extra_model_name_or_path "$EXTRA_MODEL_NAME_OR_PATH" \
    --output "$OUT_ROOT"


for CP in "${CPS[@]}"; do
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python -m experiments.src_downstream.nonhomo_RT_l2v \
        --dataset_file_path "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl" \
        --model_name "${MODEL_NAME}_cp_${CP}" \
        --instruction "$([ "$USE_INST" -eq 1 ] && echo "$INSTRUCTION" || echo "")" \
        --pooling_mode "$POOLING_MODE" \
        --max_length 512 \
        --batch_size 64 \
        --enable_bidirectional True \
        --base_model_name_or_path "$BASE_MODEL_NAME_OR_PATH" \
        --peft_model_name_or_path "$PEFT_MODEL_NAME_OR_PATH" \
        --extra_model_name_or_path "$EXTRA_MODEL_NAME_OR_PATH" "${DERMA_MODEL_PATH}/checkpoint-${CP}" \
        --output "$OUT_ROOT"
done