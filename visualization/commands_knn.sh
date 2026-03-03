## coarse grain ##
# BERTs
CUDA_VISIBLE_DEVICES=1 python visualization/knn_bert.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --model_name_or_path google-bert/bert-base-uncased \
  --output_dir /storage/BioMedNLP/llm2vec/visualization/knn_results \
  --batch_size 64

# modernBERTs
CUDA_VISIBLE_DEVICES=5 python /storage/BioMedNLP/llm2vec/visualization/knn_modernbert.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --model_name_or_path thomas-sounack/BioClinical-ModernBERT-large \
  --batch_size 64 \
  --max_length 128 \
  --attn_implementation sdpa \
  --output_dir /storage/BioMedNLP/llm2vec/visualization/knn_results/coarse

#Qwen-Embedding
CUDA_VISIBLE_DEVICES=1 python visualization/knn_qwen.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --model_name_or_path /cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B
  --output_dir /storage/BioMedNLP/llm2vec/visualization/knn_results \
  --batch_size 64 \
  --max_length 512 \
  --attn_implementation sdpa




# baseline-8B
CUDA_VISIBLE_DEVICES=6,7  python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_ResCrossAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-768_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16/checkpoint-160 \
  --instruction "Given a dermatology text describing exactly one condition among: proliferations, hair diseases,  inflammatory diseases, nail diseases, exogenous diseases, hereditary diseases and reaction-patterns-descriptive terms encode the text into an embedding that maximizes separability across these classes. Focus only on diagnosis-defining cues (lesion morphology, color, surface texture, vascularity/bleeding tendency, growth pattern, typical location, and other distinctive terms). Ignore writing style, hedging/uncertainty, patient-specific incidental context, and generic treatment wording. If multiple conditions are mentioned, encode only the single most strongly supported condition." \
  --batch_size 64 \
  --pooling_mode latent_pooling \
  --output_file /storage/BioMedNLP/llm2vec/visualization/knn_results/debug/Derml2v-8B_MixCSE_ResCrossAttn/instV3/knn_results.json

# Derm_MixCSE finetune
CUDA_VISIBLE_DEVICES=0,1  python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE/checkpoint-180 \
  --instruction "Given a dermatology sentence describing one of the following seven skin-related diseases: proliferations, hair diseases, inflammatory diseases, nail diseases, exogenous diseases, hereditary diseases, reaction-patterns-descriptive terms. Embedding should highlight the characteristics of this disease." \
  --batch_size 64

# Derm_MixCSE_ResAttn finetune
CUDA_VISIBLE_DEVICES=0,2  python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db \
  --pooling_mode latent_pooling \
  --instruction "" \
  --batch_size 64 \

# Derml2v-1.3B(MixCSE_ResAttn_DermVariantsV2)
CUDA_VISIBLE_DEVICES=0,2 python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --model_name_or_path /cache/transformers_cache/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/a4b76938edbf571ea7d7d9904861cbdca08809b4 \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp/snapshots/eb4ee4c1f922be3c5961d26eb954d0755aa9b77c \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp-supervised/snapshots/a5943d406c6b016fef3f07906aac183cf1a0b47d \
  --pooling_mode latent_pooling \
  --instruction "Given a dermatology sentence describing one of the following seven skin-related diseases: proliferations, hair diseases, inflammatory diseases, nail diseases, exogenous diseases, hereditary diseases, reaction-patterns-descriptive terms. Embedding should highlight the characteristics of this disease." \
  --batch_size 64 \
  --output_file /storage/BioMedNLP/llm2vec/visualization/knn_results/coarse/Derml2v-1p3B/isntV3/knn_results_cp-0.json

## fine-grained ##
# Classical BERT
CUDA_VISIBLE_DEVICES=1 python visualization/knn_bert.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --model_name_or_path google-bert/bert-base-uncased \
  --output_dir /storage/BioMedNLP/llm2vec/visualization/knn_results \
  --batch_size 64

# gpt
python /storage/BioMedNLP/llm2vec/visualization/knn_gpt2.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --model_name_or_path openai-community/gpt2 \
  --batch_size 64 \
  --max_length 128

# modernBERTs
python /storage/BioMedNLP/llm2vec/visualization/knn_modernbert.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --model_name_or_path thomas-sounack/BioClinical-ModernBERT-large \
  --batch_size 32 \
  --max_length 128 \
  --attn_implementation sdpa

# qwen-embedding
CUDA_VISIBLE_DEVICES=1 python visualization/knn_qwen.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv\
  --model_name_or_path /cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B
  --output_dir /storage/BioMedNLP/llm2vec/visualization/knn_results \
  --batch_size 64 \
  --max_length 512 \
  --attn_implementation sdpa

# Derml2v-8B
CUDA_VISIBLE_DEVICES=6,7  python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE/checkpoint-180 \
  --instruction "Given a dermatology text describing exactly one condition among: seborrheic keratosis, angioma, verruca vulgaris, keloid, pyogenic granuloma, dermatofibroma, milia, amyloidosis, epidermal nevus, syringoma, encode the text into an embedding that maximizes separability across these classes. Focus only on diagnosis-defining cues (lesion morphology, color, surface texture, vascularity/bleeding tendency, growth pattern, typical location, and other distinctive terms). Ignore writing style, hedging/uncertainty, patient-specific incidental context, and generic treatment wording. If multiple conditions are mentioned, encode only the single most strongly supported condition." \
  --batch_size 64

# Derml2v-1.3B
CUDA_VISIBLE_DEVICES=6,7  python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --model_name_or_path /cache/transformers_cache/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/a4b76938edbf571ea7d7d9904861cbdca08809b4 \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp/snapshots/eb4ee4c1f922be3c5961d26eb954d0755aa9b77c \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp-supervised/snapshots/a5943d406c6b016fef3f07906aac183cf1a0b47d /storage/BioMedNLP/llm2vec/output/Llama32_1p3b_mntp-supervised/withEval_QAx10_MixCSE_ResCrossAttn_DermData2/DermVariants_train_m-Sheared-LLaMA-1___3B_p-latent_pooling_b-1024_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16/checkpoint-725 \
  --instruction "" \
  --batch_size 64

# baseline
CUDA_VISIBLE_DEVICES=1,2  python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db \
  --instruction "" \
  --batch_size 64

# Derml2v-1.3B(MixCSE_ResAttn_DermVariantsV2)
CUDA_VISIBLE_DEVICES=1,2  python -m visualization.knn_l2v \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv  \
  --model_name_or_path /cache/transformers_cache/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/a4b76938edbf571ea7d7d9904861cbdca08809b4 \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp/snapshots/eb4ee4c1f922be3c5961d26eb954d0755aa9b77c \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp-supervised/snapshots/a5943d406c6b016fef3f07906aac183cf1a0b47d \
  --pooling_mode latent_pooling \
  --instruction "" \
  --batch_size 64 \
  --output_file /storage/BioMedNLP/llm2vec/visualization/knn_results/fine/Derml2v-1p3B/woinst/knn_results_cp-0.json
