# MODELS=(
#   "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
#   "emilyalsentzer/Bio_ClinicalBERT"
#   "medicalai/ClinicalBERT"
#   "michiyasunaga/BioLinkBERT-large"
# )

# for MODEL in "${MODELS[@]}"; do
#   CUDA_VISIBLE_DEVICES=1 python visualization/knn_bert.py \
#     --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
#     --model_name_or_path "${MODEL}" \
#     --output_dir /storage/BioMedNLP/llm2vec/visualization/knn_results \
#     --batch_size 64
# done

MODELS=(
    "Simonlee711/Clinical_ModernBERT"
    "thomas-sounack/BioClinical-ModernBERT-large"
)

for MODEL in "${MODELS[@]}"; do
    CUDA_VISIBLE_DEVICES=5 python /storage/BioMedNLP/llm2vec/visualization/knn_modernbert.py \
        --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
        --model_name_or_path "${MODEL}" \
        --batch_size 64 \
        --max_length 128 \
        --attn_implementation sdpa \
        --output_dir /storage/BioMedNLP/llm2vec/visualization/knn_results/coarse
done
