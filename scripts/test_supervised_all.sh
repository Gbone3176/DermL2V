#!/usr/bin/env bash
# Run experiments.test_supervised across checkpoints by cloning a user-provided
# JSON config and updating every "checkpoint-<step>" occurrence.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_json>" >&2
  echo "Optional env vars: CUDA_VISIBLE_DEVICES_VALUE, PYTHON_BIN, MODULE, START, END, STEP" >&2
  exit 1
fi

BASE_CONFIG=$1
shift || true

if [[ ! -f $BASE_CONFIG ]]; then
  echo "Config file not found: $BASE_CONFIG" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES_VALUE=${CUDA_VISIBLE_DEVICES_VALUE:-1}
PYTHON_BIN=${PYTHON_BIN:-python}
MODULE=${MODULE:-experiments.test_supervised}
START=${START:-10}
END=${END:-220}
STEP=${STEP:-10}

run_step() {
  local ckpt=$1
  local tmp_json
  tmp_json=$(mktemp "test_supervised_ckpt${ckpt}_XXXX.json")

  "$PYTHON_BIN" - <<'PY' "$BASE_CONFIG" "$tmp_json" "$ckpt"
import json
import re
import sys
from pathlib import Path

src, dst, step = sys.argv[1:4]
step = int(step)
text = Path(src).read_text(encoding="utf-8")
text = re.sub(r"checkpoint-\d+", f"checkpoint-{step}", text)
Path(dst).write_text(text, encoding="utf-8")
json.loads(Path(dst).read_text(encoding="utf-8"))
PY

  echo "\n>>> Running checkpoint-${ckpt}"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    "$PYTHON_BIN" -m "$MODULE" "$tmp_json"

  rm -f "$tmp_json"
}

for ckpt in $(seq "$START" "$STEP" "$END"); do
    run_step "$ckpt"
done
