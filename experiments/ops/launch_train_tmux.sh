#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 <config.json> [session_name] [nproc] [module]"
  exit 1
fi

CONFIG_PATH="$1"
SESSION_NAME="${2:-train}"
NPROC="${3:-1}"
MODULE_NAME="${4:-experiments.run_supervised_with_eval}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_ABS="$(cd "$REPO_ROOT" && python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_PATH")"

readarray -t ENV_INFO < <(
  cd "$REPO_ROOT" && python - <<'PY'
from pathlib import Path
import re

text = Path("local_info/local_path.md").read_text(encoding="utf-8")
keys = ["l2v_python", "l2v_torchrun"]
for key in keys:
    m = re.search(rf"- {key}: `([^`]+)`", text)
    if not m:
        raise SystemExit(f"Missing {key} in local_info/local_path.md")
    print(m.group(1))
PY
)
L2V_PYTHON="${ENV_INFO[0]}"
L2V_TORCHRUN="${ENV_INFO[1]}"
CONDA_ROOT="$(cd "$(dirname "$L2V_PYTHON")/../../.." && pwd)"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"

RESOLVED_OUTPUT_DIR="$(
  cd "$REPO_ROOT" && "$L2V_PYTHON" - <<'PY' "$CONFIG_ABS"
import json
import os
import sys
from llm2vec.experiment_utils import generate_experiment_id

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

model_name = cfg["model_name_or_path"]
if "/" in model_name:
    model_name = model_name.split("/")[-1]

experiment_id = generate_experiment_id(
    name=cfg["dataset_name"],
    split="train",
    model_name=model_name,
    pooling_mode=cfg["pooling_mode"],
    train_batch_size=cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
    max_seq_length=cfg["max_seq_length"],
    bidirectional=cfg["bidirectional"],
    epochs=cfg["num_train_epochs"],
    seed=cfg["seed"],
    warmup_steps=cfg["warmup_steps"],
    lr=cfg["learning_rate"],
    lora_r=cfg["lora_r"],
)
print(os.path.join(cfg["output_dir"], experiment_id))
PY
)"

mkdir -p "$REPO_ROOT/$RESOLVED_OUTPUT_DIR"

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  exit 1
fi

if [[ "$NPROC" == "1" ]]; then
  RUN_CMD="$L2V_PYTHON -m $MODULE_NAME \"$CONFIG_ABS\""
else
  RUN_CMD="$L2V_TORCHRUN --standalone --nproc_per_node=$NPROC -m $MODULE_NAME \"$CONFIG_ABS\""
fi

LOG_PATH="$REPO_ROOT/$RESOLVED_OUTPUT_DIR/train.log"

TMUX_CMD="source \"$CONDA_SH\" && conda activate llm2vec && cd \"$REPO_ROOT\" && $RUN_CMD 2>&1 | tee -a \"$LOG_PATH\""

tmux new-session -d -s "$SESSION_NAME" "bash"
tmux send-keys -t "$SESSION_NAME" "$TMUX_CMD" C-m

echo "session=$SESSION_NAME"
echo "output_dir=$REPO_ROOT/$RESOLVED_OUTPUT_DIR"
echo "train_log=$LOG_PATH"
