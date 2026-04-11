set -euo pipefail

INSTRUCTION="${INSTRUCTION:-Given a dermatologic question, return the answer that most closely corresponds to the information being asked for.}"
# INSTRUCTION="Given a question related to dermatology, retrieve the most relevant answer."
USE_INST="${USE_INST:-0}"

DOC_ADD_INST="${DOC_ADD_INST:-0}"

DEVICE_NUM="${DEVICE_NUM:-1}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/l2v/bin/python}"
RT_MODULE="${RT_MODULE:-experiments.src_downstream.rt_text.nonhomo.nonhomo_RT_l2v}"
DATASET_FILE="${DATASET_FILE:-/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl}"

DEVICE_COUNT=$(echo "$DEVICE_NUM" | awk -F',' '{print NF}')
if [ -z "${BATCH_SIZE:-}" ]; then
    if [ "$DEVICE_COUNT" -gt 1 ]; then
        echo "Warning: RT_MODULE runs as a single process and will only use one visible GPU; using single-GPU batch defaults." >&2
        BATCH_SIZE=32
    else
        BATCH_SIZE=64
    fi
fi
echo "BATCH_SIZE: $BATCH_SIZE"
echo "RT_MODULE: $RT_MODULE"

DERMA_MODEL_PATH="${DERMA_MODEL_PATH:-/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/woSlerpMixCSE_StructuredSelfAttn_aux5/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16}"
CPS=()
if [ -n "${CPS_LIST:-}" ]; then
    read -r -a CPS <<<"${CPS_LIST}"
elif [ "${ALL_CHECKPOINTS:-0}" -eq 1 ]; then
    while IFS= read -r cp; do
        CPS+=("$cp")
    done < <(
        find "${DERMA_MODEL_PATH}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
            | sed 's/^checkpoint-//' \
            | sort -n
    )
else
    for ((i=10; i<=50; i+=10)); do
        CPS+=($i)
    done
    if [ -d "${DERMA_MODEL_PATH}/checkpoint-132" ]; then
        CPS+=(132)
    fi
fi

MODEL_NAME="${MODEL_NAME_OVERRIDE:-$(basename "$(dirname "$DERMA_MODEL_PATH")")}"
if [ "$DOC_ADD_INST" -eq 1 ]; then
    MODEL_NAME="${MODEL_NAME}_DOC_ADD_INST"
fi
echo "MODEL_NAME: $MODEL_NAME"

if [ "$USE_INST" -eq 1 ]; then
    EVAL_INSTRUCTION="$INSTRUCTION"
    OUT_MODE="inst"
    echo "Using instruction during evaluation: $EVAL_INSTRUCTION"
else
    EVAL_INSTRUCTION=""
    OUT_MODE="woinst"
    echo "Not using instruction during evaluation."
fi
DOC_ADD_INST_FLAG=$([ "$DOC_ADD_INST" -eq 1 ] && echo "True" || echo "False")

POOLING_MODE=$(echo "$DERMA_MODEL_PATH" | sed -n 's/.*_p-\(.*\)_b-.*/\1/p')
echo "POOLING_MODE: $POOLING_MODE"

OUT_ROOT_BASE="${OUT_ROOT_BASE:-output/downstream/DermL2V/RT_text/nonhomo}"
OUT_ROOT="${OUT_ROOT:-${OUT_ROOT_BASE}/${MODEL_NAME}/${OUT_MODE}/}"
mkdir -p "$OUT_ROOT"

BASE_MODEL_NAME_OR_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PEFT_MODEL_NAME_OR_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291"
EXTRA_MODEL_NAME_OR_PATH="/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db"
SELFATTN_ATTN_HIDDEN_DIM=512
SELFATTN_NUM_HOPS=8
SELFATTN_OUTPUT_DROPOUT=0.0
SELFATTN_OUTPUT_NORM=layernorm
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# ck=0 直接表示使用基础模型进行评估, 固定pooling方式为mean
if [ "${SKIP_BASELINE:-0}" -ne 1 ]; then
    CUDA_VISIBLE_DEVICES="${DEVICE_NUM}" \
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    "${PYTHON_BIN}" -m "$RT_MODULE" \
        --dataset_file_path "$DATASET_FILE" \
        --model_name "${MODEL_NAME}_cp_0" \
        --instruction "$EVAL_INSTRUCTION" \
        --doc_add_instruction "$DOC_ADD_INST_FLAG" \
        --pooling_mode "mean" \
        --max_length 512 \
        --batch_size "$BATCH_SIZE" \
        --enable_bidirectional True \
        --base_model_name_or_path "$BASE_MODEL_NAME_OR_PATH" \
        --peft_model_name_or_path "$PEFT_MODEL_NAME_OR_PATH" \
        --extra_model_name_or_path "$EXTRA_MODEL_NAME_OR_PATH" \
        --output "$OUT_ROOT"
fi


for CP in "${CPS[@]}"; do
    CUDA_VISIBLE_DEVICES="${DEVICE_NUM}" \
        TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
        RAYON_NUM_THREADS="${RAYON_NUM_THREADS}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
        MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
        OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
        NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
        "${PYTHON_BIN}" -m "$RT_MODULE" \
        --dataset_file_path "$DATASET_FILE" \
        --model_name "${MODEL_NAME}_cp_${CP}" \
        --instruction "$EVAL_INSTRUCTION" \
        --doc_add_instruction "$DOC_ADD_INST_FLAG" \
        --pooling_mode "$POOLING_MODE" \
        --max_length 512 \
        --batch_size "$BATCH_SIZE" \
        --enable_bidirectional True \
        --selfattn_attn_hidden_dim "$SELFATTN_ATTN_HIDDEN_DIM" \
        --selfattn_num_hops "$SELFATTN_NUM_HOPS" \
        --selfattn_output_dropout "$SELFATTN_OUTPUT_DROPOUT" \
        --selfattn_output_norm "$SELFATTN_OUTPUT_NORM" \
        --base_model_name_or_path "$BASE_MODEL_NAME_OR_PATH" \
        --peft_model_name_or_path "$PEFT_MODEL_NAME_OR_PATH" \
        --extra_model_name_or_path "$EXTRA_MODEL_NAME_OR_PATH" "${DERMA_MODEL_PATH}/checkpoint-${CP}" \
        --output "$OUT_ROOT"
done
