CPS=(175 325 475 550 725)
OUT_ROOT="/storage/BioMedNLP/llm2vec/output/downstream/task4/Derml2v-1p3B"
mkdir -p "$OUT_ROOT"

for CP in "${CPS[@]}"; do
  CUDA_VISIBLE_DEVICES=1 python -m experiments.src_downstream.task4_l2v \
      --input "/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl" \
      --model_name "DermL2V-1.3B_MixCSE_ResAttn_cp-${CP}" \
      --pooling_mode "mean" \
      --max_length 512 \
      --batch_size 64 \
      --enable_bidirectional True \
      --base_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct" \
      --peft_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291" \
      --extra_model_name_or_path "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db" \
      --output "$OUT_ROOT"
done
