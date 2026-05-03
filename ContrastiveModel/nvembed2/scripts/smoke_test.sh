#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/storage/BioMedNLP/nvembed2_derm_ft"
cd "${ROOT_DIR}"

python -m compileall train.py src
