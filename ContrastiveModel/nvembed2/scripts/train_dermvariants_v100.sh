#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/storage/BioMedNLP/nvembed2_derm_ft"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${ROOT_DIR}"

python train.py --config "${ROOT_DIR}/configs/dermvariants_v100.json"
