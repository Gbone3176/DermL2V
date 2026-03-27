# Run from repo root:
# cd /storage/BioMedNLP/llm2vec

OUTPUT_DIR="/storage/BioMedNLP/llm2vec/output/downstream/RT_text/homo/combined"
VIS_DATASET="/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl"
DERMVARIANTS_DIR="/storage/dataset/dermatoscop/Derm1M/DermVariantsData"

# RT-BioClinicalBERT
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/l2v/bin/python -m experiments.src_downstream.rt_text.homo.homo_RT_bert \
    --model_path "emilyalsentzer/Bio_ClinicalBERT" \
    --model_name "BioClinicalBERT" \
    --vis_dataset "$VIS_DATASET" \
    --dermvariants_dir "$DERMVARIANTS_DIR" \
    --output "$OUTPUT_DIR"

# RT-BioLinkBERT
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/l2v/bin/python -m experiments.src_downstream.rt_text.homo.homo_RT_bert \
    --model_path "michiyasunaga/BioLinkBERT-large" \
    --model_name "BioLinkBERT" \
    --vis_dataset "$VIS_DATASET" \
    --dermvariants_dir "$DERMVARIANTS_DIR" \
    --output "$OUTPUT_DIR"

# RT-PubMedBERT
CUDA_VISIBLE_DEVICES=2 /opt/conda/envs/l2v/bin/python -m experiments.src_downstream.rt_text.homo.homo_RT_bert \
    --model_path "NeuML/pubmedbert-base-embeddings" \
    --model_name "pubmedbert-base-embeddings" \
    --vis_dataset "$VIS_DATASET" \
    --dermvariants_dir "$DERMVARIANTS_DIR" \
    --output "$OUTPUT_DIR"

# RT-GPT2
CUDA_VISIBLE_DEVICES=3 /opt/conda/envs/l2v/bin/python -m experiments.src_downstream.rt_text.homo.homo_RT_gpt2 \
    --model_path "openai-community/gpt2" \
    --model_name "gpt2" \
    --vis_dataset "$VIS_DATASET" \
    --dermvariants_dir "$DERMVARIANTS_DIR" \
    --output "$OUTPUT_DIR"

# RT-Qwen3-Embedding
CUDA_VISIBLE_DEVICES=4 /opt/conda/envs/qwen3/bin/python -m experiments.src_downstream.rt_text.homo.homo_RT_qwen3_emb \
    --model_name_or_path "/cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B" \
    --model_name "Qwen3-Embedding-8B" \
    --vis_dataset "$VIS_DATASET" \
    --dermvariants_dir "$DERMVARIANTS_DIR" \
    --output "$OUTPUT_DIR"

# RT-Clinical-ModernBERT
CUDA_VISIBLE_DEVICES=5 /opt/conda/envs/qwen3/bin/python -m experiments.src_downstream.rt_text.homo.homo_RT_modernbert \
    --model_path "Simonlee711/Clinical_ModernBERT" \
    --model_name "Clinical_ModernBERT" \
    --vis_dataset "$VIS_DATASET" \
    --dermvariants_dir "$DERMVARIANTS_DIR" \
    --output "$OUTPUT_DIR"

# RT-BioClinical-ModernBERT-large
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/qwen3/bin/python -m experiments.src_downstream.rt_text.homo.homo_RT_modernbert \
    --model_path "OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1" \
    --model_name "BioClinical-ModernBERT-large" \
    --vis_dataset "$VIS_DATASET" \
    --dermvariants_dir "$DERMVARIANTS_DIR" \
    --output "$OUTPUT_DIR"

# RT-DermL2V (structured self-attention version only)
CUDA_VISIBLE_DEVICES=7 /opt/conda/envs/l2v/bin/python -m experiments.src_downstream.rt_text.homo.homo_RT_l2v \
    --model_name "StructuredSelfAttn_QAx10_SlerpMixCSE_query-inst_cp-50" \
    --pooling_mode "structured_selfattn" \
    --max_length 512 \
    --batch_size 8 \
    --enable_bidirectional True \
    --vis_dataset "$VIS_DATASET" \
    --dermvariants_dir "$DERMVARIANTS_DIR" \
    --base_model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
    --peft_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
    --extra_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" "/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/StructuredSelfAttn_QAx10_SlerpMixCSE_query-inst/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50" \
    --output "$OUTPUT_DIR"
