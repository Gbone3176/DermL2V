LLM2VEC_PYTHON="/opt/conda/envs/l2v/bin/python"

COARSE_INSTRUCTION="Given a dermatology sentence describing one of the following seven skin-related diseases: proliferations, hair diseases, inflammatory diseases, nail diseases, exogenous diseases, hereditary diseases, reaction-patterns-descriptive terms. Embedding should highlight the characteristics of this disease."
FINE_INSTRUCTION="Given a dermatology text describing exactly one condition among: seborrheic keratosis, angioma, verruca vulgaris, keloid, pyogenic granuloma, dermatofibroma, milia, amyloidosis, epidermal nevus, syringoma, encode the text into an embedding that maximizes separability across these classes. Focus only on diagnosis-defining cues (lesion morphology, color, surface texture, vascularity/bleeding tendency, growth pattern, typical location, and other distinctive terms). Ignore writing style, hedging/uncertainty, patient-specific incidental context, and generic treatment wording. If multiple conditions are mentioned, encode only the single most strongly supported condition."

# BERT coarse
CUDA_VISIBLE_DEVICES=4 "$LLM2VEC_PYTHON" visualization/embed_bert_dosnes.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_bert_dosnes/coarse-grained \
  --model_name_or_path michiyasunaga/BioLinkBERT-large \
  --run_name coarse \
  --pca_n_components 50 \
  --dosnes_metric cosine \
  --point_size 40 &

# BERT fine
CUDA_VISIBLE_DEVICES=5 "$LLM2VEC_PYTHON" visualization/embed_bert_dosnes.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_bert_dosnes/fine-grained \
  --model_name_or_path michiyasunaga/BioLinkBERT-large \
  --run_name fine \
  --pca_n_components 50 \
  --dosnes_metric cosine \
  --point_size 40 &

# LLM2Vec baseline coarse
CUDA_VISIBLE_DEVICES=1 "$LLM2VEC_PYTHON" -m visualization.embed_l2v_dosnes \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_l2v_baseline_dosnes/coarse-grained \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db \
  --instruction "$COARSE_INSTRUCTION" \
  --run_name baseline_coarse \
  --pca_n_components 50 \
  --enable_multiprocessing \
  --batch_size 32 \
  --point_size 40 &

# LLM2Vec baseline fine
CUDA_VISIBLE_DEVICES=2 "$LLM2VEC_PYTHON" -m visualization.embed_l2v_dosnes \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_l2v_baseline_dosnes/fine-grained \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db \
  --instruction "" \
  --run_name baseline_fine \
  --pca_n_components 50 \
  --enable_multiprocessing \
  --batch_size 32 \
  --point_size 40 &

wait

# LLM2Vec MixCSE coarse
CUDA_VISIBLE_DEVICES=1 "$LLM2VEC_PYTHON" -m visualization.embed_l2v_dosnes \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_l2v_dosnes/coarse-grained \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-5_s-42_w-0_lr-2e-05_lora_r-16/checkpoint-215/checkpoint-215 \
  --instruction "$COARSE_INSTRUCTION" \
  --run_name mixcse_coarse \
  --pca_n_components 50 \
  --enable_multiprocessing \
  --batch_size 32 \
  --point_size 40 &

# LLM2Vec MixCSE fine
CUDA_VISIBLE_DEVICES=2 "$LLM2VEC_PYTHON" -m visualization.embed_l2v_dosnes \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L4/level4_proliferations_balancedsample_count200.csv \
  --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_l2v_dosnes/fine-grained \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-5_s-42_w-0_lr-2e-05_lora_r-16/checkpoint-215/checkpoint-215 \
  --instruction "$FINE_INSTRUCTION" \
  --run_name mixcse_fine \
  --pca_n_components 50 \
  --enable_multiprocessing \
  --batch_size 32 \
  --point_size 40

wait
