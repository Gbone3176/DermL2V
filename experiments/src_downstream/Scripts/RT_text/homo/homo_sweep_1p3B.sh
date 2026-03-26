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
      --base_model_name_or_path "/cache/transformers_cache/models--princeton-nlp--Sheared-LLaMA-1.3B/snapshots/a4b76938edbf571ea7d7d9904861cbdca08809b4" \
      --peft_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp/snapshots/eb4ee4c1f922be3c5961d26eb954d0755aa9b77c" \
      --extra_model_name_or_path "/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp-supervised/snapshots/a5943d406c6b016fef3f07906aac183cf1a0b47d" "/storage/BioMedNLP/llm2vec/output/Llama32_1p3b_mntp-supervised/withEval_QAx10_MixCSE_ResCrossAttn_DermData2/DermVariants_train_m-Sheared-LLaMA-1___3B_p-latent_pooling_b-1024_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16/checkpoint-${CP}" \
      --output "$OUT_ROOT"
done
