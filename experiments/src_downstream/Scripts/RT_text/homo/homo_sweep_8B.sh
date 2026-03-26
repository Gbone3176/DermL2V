DEVICE_NUM=5

DATASET_FILE_PATH="/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl"
BASE_MODEL_NAME_OR_PATH="/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct"
PEFT_MODEL_NAME_OR_PATH="/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
EXTRA_MODEL_NAME_OR_PATH="/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"

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
MODEL_NAME="DermL2V-8B_MixCSE_DataV1"
CPS=()
for ((i=5; i<=225; i+=5)); do
    CPS+=($i)
done

# DERMA_MODEL_PATH="/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_ResCrossAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-768_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16"
# MODEL_NAME="Derml2v-8B_Baseline_MixCSE_ResCrossAttn_DataV1"
# CPS=()
# for ((i=0; i<=200; i+=20)); do
#     CPS+=($i)
# done

POOLING_MODE=$(echo "$DERMA_MODEL_PATH" | sed -n 's/.*_p-\(.*\)_b-.*/\1/p')
echo "POOLING_MODE: $POOLING_MODE"

OUT_ROOT="/storage/BioMedNLP/llm2vec/output/downstream/RT_text/homo/${MODEL_NAME}"
mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python -m experiments.src_downstream.rt_text.homo.homo_RT_l2v \
    --input "$DATASET_FILE_PATH" \
    --model_name "${MODEL_NAME}_cp-0" \
    --pooling_mode "$POOLING_MODE" \
    --max_length 512 \
    --batch_size 64 \
    --enable_bidirectional True \
    --base_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
    --peft_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
    --extra_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" \
    --output "$OUT_ROOT"

for CP in "${CPS[@]}"; do
    CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python -m experiments.src_downstream.rt_text.homo.homo_RT_l2v \
        --input "$DATASET_FILE_PATH" \
        --model_name "${MODEL_NAME}_cp-${CP}" \
        --pooling_mode "$POOLING_MODE" \
        --max_length 512 \
        --batch_size 64 \
        --enable_bidirectional True \
        --base_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
        --peft_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
        --extra_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" \
        --output "$OUT_ROOT"
done
