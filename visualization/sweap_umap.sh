# Bert grid search (umap_n_neighbors, umap_min_dist)
#
# You can override the sweep values via environment variables:
#   UMAP_N_NEIGHBORS_VALUES="5 10 15 30"
#   UMAP_MIN_DIST_VALUES="0.0 0.05 0.1 0.2"

# UMAP_N_NEIGHBORS_VALUES="${UMAP_N_NEIGHBORS_VALUES:-"5 10 15 30"}"
# UMAP_MIN_DIST_VALUES="${UMAP_MIN_DIST_VALUES:-"0.0 0.05 0.1 0.2"}"
# BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-32}"

# read -r -a UMAP_N_NEIGHBORS_LIST <<< "${UMAP_N_NEIGHBORS_VALUES}"
# read -r -a UMAP_MIN_DIST_LIST <<< "${UMAP_MIN_DIST_VALUES}"

# for nn in "${UMAP_N_NEIGHBORS_LIST[@]}"; do
#   for md in "${UMAP_MIN_DIST_LIST[@]}"; do
#     echo "[BERT UMAP] n_neighbors=${nn} min_dist=${md}"
#     python visualization/embed_bert_umap.py \
#       --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv\
#       --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_bert_umap/pca \
#       --model_name_or_path michiyasunaga/BioLinkBERT-large \
#       --batch_size "${BERT_BATCH_SIZE}" \
#       --umap_n_neighbors "${nn}" \
#       --umap_min_dist "${md}" \
#       --pca_n_components 50 \
#       --point_size 40 \
#       --umap_metric cosine
#   done
# done


# L2V grid search (n_neighbors, min_dist)
L2V_N_NEIGHBORS_VALUES="${L2V_N_NEIGHBORS_VALUES:-"30"}"
L2V_MIN_DIST_VALUES="${L2V_MIN_DIST_VALUES:-"0.0 0.05 0.1 0.2 0.3"}"
L2V_BATCH_SIZE="${L2V_BATCH_SIZE:-32}"
L2V_OUTPUT_FOLDER="${L2V_OUTPUT_FOLDER:-/storage/BioMedNLP/llm2vec/visualization/embed_l2v_baseline_umap/coarse-grained}"

read -r -a L2V_N_NEIGHBORS_LIST <<< "${L2V_N_NEIGHBORS_VALUES}"
read -r -a L2V_MIN_DIST_LIST <<< "${L2V_MIN_DIST_VALUES}"

for nn in "${L2V_N_NEIGHBORS_LIST[@]}"; do
  for md in "${L2V_MIN_DIST_LIST[@]}"; do
    echo "[L2V UMAP] n_neighbors=${nn} min_dist=${md}"
    CUDA_VISIBLE_DEVICES=2,4,5,6 python -m visualization.embed_l2v_umap \
      --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
      --output_folder "${L2V_OUTPUT_FOLDER}" \
      --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
      --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
      --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db \
      --instruction "Given a dermatology text describing exactly one condition among: seborrheic keratosis, angioma, verruca vulgaris, keloid, pyogenic granuloma, dermatofibroma, milia, amyloidosis, epidermal nevus, syringoma, encode the text into an embedding that maximizes separability across these classes. Focus only on diagnosis-defining cues (lesion morphology, color, surface texture, vascularity/bleeding tendency, growth pattern, typical location, and other distinctive terms). Ignore writing style, hedging/uncertainty, patient-specific incidental context, and generic treatment wording. If multiple conditions are mentioned, encode only the single most strongly supported condition." \
      --pca_n_components 50 \
      --enable_multiprocessing \
      --batch_size "${L2V_BATCH_SIZE}" \
      --n_neighbors "${nn}" \
      --min_dist "${md}" \
      --point_size 40
  done
done
