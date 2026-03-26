#!/usr/bin/env bash
# Run experiments.test_supervised across checkpoints 5..150 (step 5)
# by materializing the full JSON config inline for each step.

set -euo pipefail

CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
MODULE=${MODULE:-experiments.test_supervised}
START=${START:-25}
END=${END:-800}
STEP=${STEP:-50}

SUFFIX="withEval_QAx10_MixCSE_ResCrossAttn_DermData2/DermVariants_train_m-Sheared-LLaMA-1___3B_p-latent_pooling_b-1024_l-512_bidirectional-True_e-5_s-42_w-100_lr-2e-05_lora_r-16/checkpoint-__CKPT__"

run_step() {
    local ckpt=$1
    local tmp_json
    tmp_json=$(mktemp "test_supervised_ckpt${ckpt}_XXXX.json")

    cat <<EOF > "$tmp_json"
{
  "model_name_or_path": "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/cache/modelscope/hub/models/princeton-nlp/Sheared-LLaMA-1___3B",
  "peft_model_name_or_path": "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/cache/huggingface/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp/snapshots/eb4ee4c1f922be3c5961d26eb954d0755aa9b77c",
  "extra_model_name_or_path" : [
    "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/cache/huggingface/hub/models--McGill-NLP--LLM2Vec-Sheared-LLaMA-mntp-supervised/snapshots/a5943d406c6b016fef3f07906aac183cf1a0b47d",
    "/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec/output/Llama32_1p3b_mntp-supervised/${SUFFIX}"
  ],
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