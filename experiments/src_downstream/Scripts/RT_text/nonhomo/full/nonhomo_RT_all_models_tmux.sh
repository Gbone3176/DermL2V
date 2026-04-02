#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="${SESSION_NAME:-nonhomo_full_eval}"
ROOT_DIR="/storage/BioMedNLP/llm2vec"
SCRIPT_DIR="${ROOT_DIR}/experiments/src_downstream/Scripts/RT_text/nonhomo/full"
LOG_DIR="${ROOT_DIR}/output/downstream/DermL2V/RT_text/nonhomo_full/logs"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    tmux kill-session -t "${SESSION_NAME}"
fi

tmux new-session -d -s "${SESSION_NAME}" -n evals
tmux setw -t "${SESSION_NAME}" remain-on-exit on

tmux send-keys -t "${SESSION_NAME}:0.0" "/bin/bash --noprofile --norc -lc 'cd ${ROOT_DIR} && CUDA_DEVICE=0 bash ${SCRIPT_DIR}/nonhomo_RT_bert_full.sh |& tee ${LOG_DIR}/bert.log'" C-m

tmux new-window -t "${SESSION_NAME}" -n qwen "/bin/bash --noprofile --norc -lc 'cd ${ROOT_DIR} && CUDA_DEVICE=1 bash ${SCRIPT_DIR}/nonhomo_RT_qwen_full.sh |& tee ${LOG_DIR}/qwen.log'"
tmux new-window -t "${SESSION_NAME}" -n modernbert "/bin/bash --noprofile --norc -lc 'cd ${ROOT_DIR} && CUDA_DEVICE=2 bash ${SCRIPT_DIR}/nonhomo_RT_modernbert_full.sh |& tee ${LOG_DIR}/modernbert.log'"
tmux new-window -t "${SESSION_NAME}" -n gpt2 "/bin/bash --noprofile --norc -lc 'cd ${ROOT_DIR} && CUDA_DEVICE=3 bash ${SCRIPT_DIR}/nonhomo_RT_gpt2_full.sh |& tee ${LOG_DIR}/gpt2.log'"
tmux new-window -t "${SESSION_NAME}" -n nvembed "/bin/bash --noprofile --norc -lc 'cd ${ROOT_DIR} && CUDA_DEVICE=4 bash ${SCRIPT_DIR}/nonhomo_RT_nvembed_full.sh |& tee ${LOG_DIR}/nvembed.log'"
tmux new-window -t "${SESSION_NAME}" -n derml2v-mean "/bin/bash --noprofile --norc -lc 'cd ${ROOT_DIR} && CUDA_DEVICE=5 bash ${SCRIPT_DIR}/nonhomo_RT_derml2v_mean_full.sh |& tee ${LOG_DIR}/derml2v_mean.log'"
tmux new-window -t "${SESSION_NAME}" -n derml2v-sa "/bin/bash --noprofile --norc -lc 'cd ${ROOT_DIR} && CUDA_DEVICE=6 bash ${SCRIPT_DIR}/nonhomo_RT_derml2v_selfattn_full.sh |& tee ${LOG_DIR}/derml2v_selfattn.log'"
tmux new-window -t "${SESSION_NAME}" -n reserved "/bin/bash --noprofile --norc -lc 'cd ${ROOT_DIR} && echo Reserved window on GPU 7 for reruns or monitoring | tee ${LOG_DIR}/reserved.log && tail -f /dev/null'"

tmux display-message -t "${SESSION_NAME}:0" "Started ${SESSION_NAME} with multiple windows."
echo "tmux session started: ${SESSION_NAME}"
