# 在11服务器上运行执行这个任务!!

# task4-bert
CUDA_VISIBLE_DEVICES=5 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/task4_bert.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_path "thomas-sounack/BioClinical-ModernBERT-large" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/task4/"

# task4-gpt2
CUDA_VISIBLE_DEVICES=5 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/task4_gpt2.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_path "openai-community/gpt2" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/task4/"

# task4-qwen3-embedding
CUDA_VISIBLE_DEVICES=5 python /storage/BioMedNLP/llm2vec/experiments/src_downstream/task4_qwen3_emb.py \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/task4/"

# task4-LLM-based: LLM2VEC(Llama3.1)
CUDA_VISIBLE_DEVICES=5 python -m experiments.src_downstream.task4_l2v \
    --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
    --model_name "LLM2VEC4Derm(Llama3.1-8B-Instruct) wo MixCSE" \
    --pooling_mode "mean" \
    --max_length 512 \
    --batch_size 64 \
    --enable_bidirectional True \
    --base_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
    --peft_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
    --extra_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" \
    --output "/storage/BioMedNLP/llm2vec/output/downstream/task4/"