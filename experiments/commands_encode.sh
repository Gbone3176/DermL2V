    
###################################################### encode pretrain data ###################################################################

CUDA_VISIBLE_DEVICES=6 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/Derm1M_v2_pretrain.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/Derm1M_v2_pretrain" \
    --batch_size 96 \
    --device cuda \
    --instruction "" \
    --num_shards 6 \
    --shard_id 0 

CUDA_VISIBLE_DEVICES=1 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/Derm1M_v2_pretrain.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/Derm1M_v2_pretrain" \
    --batch_size 96 \
    --device cuda \
    --instruction "" \
    --num_shards 6 \
    --shard_id 1

CUDA_VISIBLE_DEVICES=2 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/Derm1M_v2_pretrain.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/Derm1M_v2_pretrain" \
    --batch_size 96 \
    --device cuda \
    --instruction "" \
    --num_shards 6 \
    --shard_id 2 

CUDA_VISIBLE_DEVICES=3 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/Derm1M_v2_pretrain.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/Derm1M_v2_pretrain" \
    --batch_size 96 \
    --device cuda \
    --instruction "" \
    --num_shards 6 \
    --shard_id 3

CUDA_VISIBLE_DEVICES=4 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/Derm1M_v2_pretrain.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/Derm1M_v2_pretrain" \
    --batch_size 96 \
    --device cuda \
    --instruction "" \
    --num_shards 6 \
    --shard_id 4

CUDA_VISIBLE_DEVICES=5 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/Derm1M_v2_pretrain.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/Derm1M_v2_pretrain" \
    --batch_size 96 \
    --device cuda \
    --instruction "" \
    --num_shards 6 \
    --shard_id 5
    
############################################################# encode evaluate data ###################################################################

CUDA_VISIBLE_DEVICES=6 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/Derm1M_v2_validation.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/Derm1M_v2_validation" \
    --batch_size 64 \
    --device cuda \
    --instruction "" \
    --num_shards 2 \
    --shard_id 0 

CUDA_VISIBLE_DEVICES=7 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/Derm1M_v2_validation.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/Derm1M_v2_validation" \
    --batch_size 64 \
    --device cuda \
    --instruction "" \
    --num_shards 2 \
    --shard_id 1

############################################################# encode skincap ###################################################################

CUDA_VISIBLE_DEVICES=3 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp1_MultiModelRetrieval/skincap_3989.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/skincap_3989" \
    --batch_size 64 \
    --device cuda \
    --instruction "" \
    --num_shards 2 \
    --shard_id 0

CUDA_VISIBLE_DEVICES=4 python -m experiments.encode \
    --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
    --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-160" \
    --input_file "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp1_MultiModelRetrieval/skincap_3989.jsonl" \
    --output_dir "/storage/dataset/dermatoscop/Derm1M/MultiModelRetrieval/text_features/Derml2v_MixCSE_dataV2_cp160/skincap_3989" \
    --batch_size 64 \
    --device cuda \
    --instruction "" \
    --num_shards 2 \
    --shard_id 1 


####################### encode_toy ##################
# LLM2Vec-Derm
CUDA_VISIBLE_DEVICES=2 python -m experiments.encode_l2v_similarity \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-5_s-42_w-0_lr-2e-05_lora_r-16/checkpoint-45"  \
  --sentence1 "The poster, male, has been experiencing symptoms for a week, including itching and redness in the coronal sulcus. He is worried about sexually transmitted diseases (STDs) but has tested negative for gonorrhea. Despite using antifungal cream (Baidubang), his symptoms have not improved. He mentions a recent sexual encounter with condom use and is seeking advice on his condition and how to treat it. Users in the forum suggest that the symptoms might indicate inflammation or an infection in the coronal sulcus. Potential diagnoses mentioned include candidal balanoposthitis and a general fungal infection. Treatments suggested by users include application of erythromycin ointment, ketoconazole, Dakin solution, and Econazole nitrate cream. It is also mentioned that using condoms still leaves a small risk of STD transmission if not all body fluids are isolated. One user suggests considering a drug eruption as a possibility." \
  --sentence2 "The poster, male, has been experiencing symptoms for a week, including itching and redness in the coronal sulcus. He is worried about sexually transmitted diseases (STDs) but has tested negative for gonorrhea. Despite using antifungal cream (Baidubang), his symptoms have not improved. He mentions a recent sexual encounter with condom use and is seeking advice on his condition and how to treat it. Users in the forum suggest that the symptoms might indicate inflammation or an infection in the coronal sulcus. Potential diagnoses mentioned include candidal balanoposthitis and a general fungal infection. Treatments suggested by users include application of erythromycin ointment, ketoconazole, Dakin solution, and Econazole nitrate cream. It is also mentioned that using condoms still leaves a small risk of STD transmission if not all body fluids are isolated. One user suggests considering a drug eruption as a possibility." \
  --instruction "Given an Dermatology sentence, encode the text into an embedding that prioritizes image-verifiable visual attributes. Keep only information that could be confirmed from an image. Emphasize: body part or region, appearance, shape, color, texture, edges, distribution, size/extent, and count. De-emphasize or ignore: diagnosis names, causes, treatment, history, and non-visual commentary. Preserve the remaining meaning faithfully."

# LLM2Vec-basemodel
CUDA_VISIBLE_DEVICES=2 python -m experiments.encode_l2v_similarity \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path  "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" \
  --sentence1 "This is a clinical skin image, diagnosed as urticaria hives." \
  --sentence2 "This is a clinical skin image, diagnosed as systemic lupus erythematosus."

# BERT
CUDA_VISIBLE_DEVICES=2 python -m experiments.encode_bert_similarity \
  --model_name_or_path michiyasunaga/BioLinkBERT-large \
  --sentence1 "This is a clinical skin image, diagnosed as urticaria hives." \
  --sentence2 "This is a clinical skin image, diagnosed as systemic lupus erythematosus."