# RT-bert
CUDA_VISIBLE_DEVICES=2 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/RT_bert.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl" \
    --model_path "emilyalsentzer/Bio_ClinicalBERT" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/RT_text/"


# RT-ModernBert
CUDA_VISIBLE_DEVICES=2 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/RT_modernbert.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl" \
    --model_path "thomas-sounack/BioClinical-ModernBERT-large" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/RT_text/"



# RT-gpt2
CUDA_VISIBLE_DEVICES=5 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/RT_gpt2.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl" \
    --model_path "openai-community/gpt2" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/RT_text/"




# RT-qwen3-embedding
CUDA_VISIBLE_DEVICES=4 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/RT_qwen.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl" \
    --model_name_or_path "/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/RT_text/Qwen3-Embedding"

# 测试: RT-Derml2v  query with instruction, doc without instruction
CUDA_VISIBLE_DEVICES=6,7 python -m experiments.src_downstream.RT_l2v \
    --instruction "Given a question related to dermatology, retrieve the most relevant answer." \
    --dataset_file_path "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl" \
    --model_name "Derml2v-8B_MixCSE_DataV2_inst_cp_180" \
    --pooling_mode "mean" \
    --max_length 512 \
    --batch_size 64 \
    --enable_bidirectional True \
    --base_model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    --peft_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
    --extra_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1024_l-512_bidirectional-True_e-2_s-42_w-15_lr-2e-05_lora_r-16/checkpoint-180" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/debug/"


# RT-Derml2v with instruction
CUDA_VISIBLE_DEVICES=6,7 python -m experiments.src_downstream.nonhomo_RT_l2v_inst \
    --instruction "Given a question related to dermatology, retrieve the most relevant answer." \
    --dataset_file_path "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl" \
    --model_name "Llama-31-8B_baseline_MixCSE_iqid" \
    --pooling_mode "mean" \
    --max_length 512 \
    --batch_size 64 \
    --enable_bidirectional True \
    --base_model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    --peft_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
    --extra_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1536_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-140" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/debug/RT_homo/instV0/"

# RT-DermL2V-1.3B
CUDA_VISIBLE_DEVICES=1,2 python -m experiments.src_downstream.RT_l2v \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_name "DermL2V-1.3B_MixCSE_ResAttn_cp-0" \
    --pooling_mode "mean" \
    --max_length 512 \
    --batch_size 64 \
    --enable_bidirectional True \
    --base_model_name_or_path "/cache/transformers_cache/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/a4b76938edbf571ea7d7d9904861cbdca08809b4" \
    --peft_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp/snapshots/eb4ee4c1f922be3c5961d26eb954d0755aa9b77c" \
    --extra_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp-supervised/snapshots/a5943d406c6b016fef3f07906aac183cf1a0b47d" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/task4/Derml2v-1p3B/"