python visualization/embed_bert_dosnes.py \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_bert_dosnes/coarse-grained \
  --model_name_or_path michiyasunaga/BioLinkBERT-large \
  --pca_n_components 50 \
  --dosnes_metric cosine \
  --point_size 40

CUDA_VISIBLE_DEVICES=1 python -m visualization.embed_l2v_dosnes \
  --input_file /storage/dataset/dermatoscop/Derm1M/embedding_space_vis/L1/test.csv \
  --output_folder /storage/BioMedNLP/llm2vec/visualization/embed_l2v_dosnes/coarse-grained \
  --model_name_or_path /cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291 \
  --extra_model_name_or_path /cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db \
  --instruction "Given a dermatology text describing exactly one condition among: proliferations, hair diseases, inflammatory diseases, nail diseases, exogenous diseases, hereditary diseases and reaction-patterns-descriptive terms encode the text into an embedding that maximizes separability across these classes. Focus only on diagnosis-defining cues (lesion morphology, color, surface texture, vascularity/bleeding tendency, growth pattern, typical location, and other distinctive terms). Ignore writing style, hedging/uncertainty, patient-specific incidental context, and generic treatment wording. If multiple conditions are mentioned, encode only the single most strongly supported condition." \
  --pca_n_components 50 \
  --enable_multiprocessing \
  --batch_size 32 \
  --point_size 40
