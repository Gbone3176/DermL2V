#!/usr/bin/env bash
# Run experiments.test_supervised across checkpoints 5..150 (step 5)
# by materializing the full JSON config inline for each step.

set -euo pipefail

CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
MODULE=${MODULE:-experiments.test_supervised}
START=${START:-20}
END=${END:-325}
STEP=${STEP:-40}

SUFFIX="DermVariants/withEval_QAx10_MixCSE_ResCrossAttn/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-latent_pooling_b-768_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16/checkpoint-__CKPT__"

run_step() {
    local ckpt=$1
    local tmp_json
    tmp_json=$(mktemp "test_supervised_ckpt${ckpt}_XXXX.json")

    cat <<EOF > "$tmp_json"
{
  "model_name_or_path": "/mnt/nas1/disk06/bowenguo/cache/modelscope/hub/models/LLM-Research/Meta-Llama-31-8B-Instruct",
  "peft_model_name_or_path": "/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291",
  "extra_model_name_or_path" : ["/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"],
  "dataset_name":"DermVariants",
  "split":"test",
  "dataset_file_path": "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/text-img/dermatoscop/DermVariantsData",
  "output_dir": "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec/test_results/test_Supervised/mntp-supervised/${SUFFIX}",
  "batch_size": 128,
  "top_k": 10,
  "bidirectional": true,
  "max_seq_length": 512,
  "torch_dtype": "bfloat16",
  "pooling_mode": "latent_pooling",
  "attn_implementation": "flash_attention_2",
  "dermqa_upsample_ratio": 1
}
EOF

    sed -i "s/__CKPT__/${ckpt}/g" "$tmp_json"

    echo "\n>>> Running checkpoint-${ckpt}"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
        "$PYTHON_BIN" -m "$MODULE" "$tmp_json"

    rm -f "$tmp_json"
}

for ckpt in $(seq "$START" "$STEP" "$END"); do
    run_step "$ckpt"
done