
# RT-bert
CUDA_VISIBLE_DEVICES=4 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/rt_text/homo/homo_RT_bert.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_path "NeuML/pubmedbert-base-embeddings" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/VisMatch"


# RT-ModernBert(使用qwen3环境)
CUDA_VISIBLE_DEVICES=4 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/rt_text/homo/homo_RT_modernbert.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_path "OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/VisMatch"


# RT-gpt2
CUDA_VISIBLE_DEVICES=5 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/rt_text/homo/homo_RT_gpt2.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_path "openai-community/gpt2" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/VisMatch"


# RT-qwen3-embedding
CUDA_VISIBLE_DEVICES=4 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/rt_text/homo/homo_RT_qwen3_emb.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/VisMatch"


##Derml2v_baseline_DataV1
MODEL_NAME=Derml2v_baseline_DataV1
DermL2V_ADAPTER_PATH=/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-3_s-42_w-0_lr-2e-05_lora_r-16/checkpoint-120
POOLING=mean

##Derml2v_baseline_MIXCSE_DataV1
MODEL_NAME=Derml2v_baseline_MIXCSE_DataV1
POOLING=mean
DermL2V_ADAPTER_PATH=/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-768_l-512_bidirectional-True_e-5_s-42_w-0_lr-2e-05_lora_r-16/checkpoint-210

##Derml2v_baseline_MIXCSE_ResattnDataV1
DermL2V_ADAPTER_PATH=/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_ResCrossAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-768_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16/checkpoint-280
MODEL_NAME=Derml2v_baseline_MIXCSE_ResattnDataV1
POOLING=latent_pooling

# RT-Derml2v
CUDA_VISIBLE_DEVICES=3 python -m experiments.src_downstream.rt_text.homo.homo_RT_l2v \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_name "${MODEL_NAME}" \
    --pooling_mode "${POOLING}" \
    --max_length 512 \
    --batch_size 64 \
    --enable_bidirectional True \
    --base_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
    --peft_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
    --extra_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"\
    --output "/storage/BioMedNLP/llm2vec/output/downstream/RT_text/homo/DermL2V-8B"

# RT-DermL2V-1.3B
CUDA_VISIBLE_DEVICES=1,2 python -m experiments.src_downstream.rt_text.homo.homo_RT_l2v \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_name "DermL2V-1.3B_MixCSE_ResAttn_cp-0" \
    --pooling_mode "mean" \
    --max_length 512 \
    --batch_size 64 \
    --enable_bidirectional True \
    --base_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
    --peft_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
    --extra_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/RT_text/Derml2v-1p3B/"