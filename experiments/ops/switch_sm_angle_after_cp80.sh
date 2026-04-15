#!/usr/bin/env bash
set -euo pipefail

ROOT="/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec"
TORCHRUN_BIN="/home/gbw_21307130160/anaconda3/envs/llm2vec/bin/torchrun"

CURRENT_SESSION="SM_AngleLoss_4g_20260413"
NEXT_SESSION="SM_AngleLoss_aw0p2_4g_20260413"

CURRENT_DIR="$ROOT/output/Llama31_8b_mntp-supervised/DermL2V/SM_AngleLoss/SlerpMixCSE_k16_cw1p0_aw0p02"
CURRENT_CFG="$ROOT/train_configs/SM_AngleLoss/Llama31-8b-inst-mntp-supervised_SM_AngleLoss@DermVariantsSFT.json"
NEXT_CFG="$ROOT/train_configs/SM_AngleLoss/Llama31-8b-inst-mntp-supervised_SM_AngleLoss_aw0p2@DermVariantsSFT.json"
NEXT_LOG="$ROOT/output/Llama31_8b_mntp-supervised/DermL2V/SM_AngleLoss/SlerpMixCSE_k16_cw1p0_aw0p2/launch_$(date +%Y%m%d_%H%M).log"

log() {
    echo "[$(date '+%F %T')] $*"
}

current_pids() {
    pgrep -f "$CURRENT_CFG" || true
}

log "waiting_for_checkpoint_80:$CURRENT_DIR"
while ! find "$CURRENT_DIR" -path "*/checkpoint-80/trainer_state.json" | grep -q checkpoint-80; do
    sleep 30
done

log "checkpoint_80_detected"
tmux send-keys -t "${CURRENT_SESSION}:0.0" C-c
sleep 15

if [[ -n "$(current_pids)" ]]; then
    log "tmux_ctrl_c_did_not_stop_run; sending SIGINT to matched pids"
    current_pids | xargs -r kill -INT
    sleep 15
fi

if [[ -n "$(current_pids)" ]]; then
    log "sigint_did_not_stop_run; sending SIGTERM to matched pids"
    current_pids | xargs -r kill -TERM
    sleep 15
fi

if [[ -n "$(current_pids)" ]]; then
    log "sigterm_did_not_stop_run; sending SIGKILL to matched pids"
    current_pids | xargs -r kill -KILL
fi

while [[ -n "$(current_pids)" ]]; do
    sleep 10
done

mkdir -p "$(dirname "$NEXT_LOG")"

if tmux has-session -t "$NEXT_SESSION" 2>/dev/null; then
    tmux kill-session -t "$NEXT_SESSION"
fi

log "starting_next_run:$NEXT_CFG"
tmux new-session -d -s "$NEXT_SESSION" \
    "cd $ROOT && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export PYTHONPATH=$ROOT:\${PYTHONPATH} && $TORCHRUN_BIN --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval $NEXT_CFG 2>&1 | tee -a $NEXT_LOG"

log "done"
