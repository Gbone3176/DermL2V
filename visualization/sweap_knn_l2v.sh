############################
# Derml2v-1.3B
############################

# coarsed
# CPS=(25 50 75 100 125 150 175 325 475 550 725)
# OUT_DIR="/storage/BioMedNLP/llm2vec/visualization/knn_results/coarse/Derml2v-1p3B/instV2"
# mkdir -p "$OUT_DIR"

# for CP in "${CPS[@]}"; do
#   CUDA_VISIBLE_DEVICES=1,2 python -m visualization.knn_l2v \
#     --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
#     --model_name_or_path /cache/transformers_cache/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/a4b76938edbf571ea7d7d9904861cbdca08809b4 \
#     --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp/snapshots/eb4ee4c1f922be3c5961d26eb954d0755aa9b77c \
#     --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp-supervised/snapshots/a5943d406c6b016fef3f07906aac183cf1a0b47d /storage/BioMedNLP/llm2vec/output/Llama32_1p3b_mntp-supervised/withEval_QAx10_MixCSE_ResCrossAttn_DermData2/DermVariants_train_m-Sheared-LLaMA-1___3B_p-latent_pooling_b-1024_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16/checkpoint-"$CP" \
#     --pooling_mode latent_pooling \
#     --instruction "Given a dermatology sentence describing one of the following seven skin-related diseases: proliferations, hair diseases, inflammatory diseases, nail diseases, exogenous diseases, hereditary diseases, reaction-patterns-descriptive terms. Embedding should highlight the characteristics of this disease." \
#     --batch_size 64 \
#     --output_file "$OUT_DIR/knn_results_cp-${CP}.json"
# done

# fine-grained
# CPS=(25 50 75 100 125 150 175 325 475 550 725)
# OUT_DIR="/storage/BioMedNLP/llm2vec/visualization/knn_results/fine/Derml2v-1p3B/woinst"
# mkdir -p "$OUT_DIR"


# for CP in "${CPS[@]}"; do
#   CUDA_VISIBLE_DEVICES=1,2 python -m visualization.knn_l2v \
#     --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv\
#     --model_name_or_path /cache/transformers_cache/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/a4b76938edbf571ea7d7d9904861cbdca08809b4 \
#     --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp/snapshots/eb4ee4c1f922be3c5961d26eb954d0755aa9b77c \
#     --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp-supervised/snapshots/a5943d406c6b016fef3f07906aac183cf1a0b47d /storage/BioMedNLP/llm2vec/output/Llama32_1p3b_mntp-supervised/withEval_QAx10_MixCSE_ResCrossAttn_DermData2/DermVariants_train_m-Sheared-LLaMA-1___3B_p-latent_pooling_b-1024_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16/checkpoint-"$CP" \
#     --pooling_mode latent_pooling \
#     --instruction "" \
#     --batch_size 64 \
#     --output_file "$OUT_DIR/knn_results_cp-${CP}.json"
# done





############################
# Derml2v-8B
############################

######################### coarsed-grained #########################

# OUT_DIR="/storage/BioMedNLP/llm2vec/visualization/knn_results/coarse/Derml2v-8B_MixCSE_DataV2/instV3"
# mkdir -p "$OUT_DIR"

# COARSED_INSTRUCTION="Given a dermatology text describing exactly one condition among: proliferations, hair diseases,  inflammatory diseases, nail diseases, exogenous diseases, hereditary diseases and reaction-patterns-descriptive terms encode the text into an embedding that maximizes separability across these classes. Focus only on diagnosis-defining cues (lesion morphology, color, surface texture, vascularity/bleeding tendency, growth pattern, typical location, and other distinctive terms). Ignore writing style, hedging/uncertainty, patient-specific incidental context, and generic treatment wording. If multiple conditions are mentioned, encode only the single most strongly supported condition."

# CUDA_VISIBLE_DEVICES=1  python -m visualization.knn_l2v \
#   --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
#   --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
#   --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
#   --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db \
#   --instruction "$COARSED_INSTRUCTION" \
#   --batch_size 64 \
#   --pooling_mode mean \
#   --output_file "$OUT_DIR/knn_results_cp-0.json"

# CPS=()
# for ((i=10; i<=320; i+=10)); do
#     CPS+=("$i")
# done

# for CP in "${CPS[@]}"; do
#   CUDA_VISIBLE_DEVICES=1 python -m visualization.knn_l2v \
#     --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
#     --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
#     --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
#     --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-"$CP" \
#     --pooling_mode mean \
#     --instruction "$COARSED_INSTRUCTION" \
#     --batch_size 64 \
#     --output_file "$OUT_DIR/knn_results_cp-${CP}.json"
# done


######################### fine-grained #########################

OUT_DIR="/storage/BioMedNLP/llm2vec/visualization/knn_results/fine/Derml2v-8B_MixCSE_DataV2/instV3"
mkdir -p "$OUT_DIR"

FINE_INSTRUCTION="Given a dermatology text describing exactly one condition among: seborrheic keratosis, angioma, verruca vulgaris, keloid, pyogenic granuloma, dermatofibroma, milia, amyloidosis, epidermal nevus, syringoma, encode the text into an embedding that maximizes separability across these classes. Focus only on diagnosis-defining cues (lesion morphology, color, surface texture, vascularity/bleeding tendency, growth pattern, typical location, and other distinctive terms). Ignore writing style, hedging/uncertainty, patient-specific incidental context, and generic treatment wording. If multiple conditions are mentioned, encode only the single most strongly supported condition."

CUDA_VISIBLE_DEVICES=2,3 python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db \
  --pooling_mode mean \
  --instruction "$FINE_INSTRUCTION" \
  --batch_size 64 \
  --output_file "$OUT_DIR/knn_results_cp-0.json"

CPS=()
for ((i=10; i<=320; i+=10)); do
    CPS+=("$i")
done

for CP in "${CPS[@]}"; do
  CUDA_VISIBLE_DEVICES=2,3 python -m visualization.knn_l2v \
    --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-"$CP" \
    --pooling_mode mean \
    --instruction "$FINE_INSTRUCTION" \
    --batch_size 64 \
    --output_file "$OUT_DIR/knn_results_cp-${CP}.json"
done