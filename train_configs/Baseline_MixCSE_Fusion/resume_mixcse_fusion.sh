#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-train_configs/Baseline_MixCSE_Fusion/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT_mixcse_fusion.json}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
SWANLAB_MODE_VALUE="${SWANLAB_MODE:-cloud}"
LOG_DIR="${LOG_DIR:-logs}"
CONDA_SH="${CONDA_SH:-/home/gbw_21307130160/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-llm2vec}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "${CONFIG_PATH%.json}")_resume_$(date +%Y%m%d_%H%M%S).log"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_PROCS="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export SWANLAB_MODE="$SWANLAB_MODE_VALUE"
export PYTHONUNBUFFERED=1

echo "Root dir: $ROOT_DIR"
echo "Config: $CONFIG_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE: $NUM_PROCS"
echo "SWANLAB_MODE: $SWANLAB_MODE"
echo "Log file: $LOG_FILE"

if [[ "$NUM_PROCS" -le 1 ]]; then
    RUN_CMD=(python -m experiments.run_supervised_fusion_withEval "$CONFIG_PATH")
else
    RUN_CMD=(torchrun --standalone --nproc_per_node="$NUM_PROCS" -m experiments.run_supervised_fusion_withEval "$CONFIG_PATH")
fi

printf 'Command:'
printf ' %q' "${RUN_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

"${RUN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
