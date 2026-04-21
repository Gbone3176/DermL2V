#!/usr/bin/env bash
set -euo pipefail

ROOT="/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec"

CURRENT_SESSION="l32_smsa_r3"
CURRENT_CFG_REL="train_configs/Llama31_8B_workflow_baseline/SA_SM_AngleLoss/Llama31-8b-inst-mntp-sup_SM_SA_aux0p001_aw0p02_cw1p0_gamma0p001_k128_onormln@DermVariantsSFT.json"
CURRENT_DIR="$ROOT/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM_AngleLoss/aux0p001_aw0p02_cw1p0_gamma0p001_k128_onormln"

NEXT_CFG_REL="train_configs/Llama31_8B_workflow_baseline/baseline/Llama31-8b-inst-mntp-sup@DermVariantsSFT-LLM2vecSettings.json"
NEXT_SESSION="l31_base_l2vset"

log() {
    echo "[$(date '+%F %T')] $*"
}

current_pids() {
    pgrep -f "$CURRENT_CFG_REL" || true
}

log "waiting_for_checkpoint_80:$CURRENT_DIR"
while ! find "$CURRENT_DIR" -path "*/checkpoint-80/trainer_state.json" | grep -q "checkpoint-80"; do
    sleep 30
done

log "checkpoint_80_detected"

if tmux has-session -t "$CURRENT_SESSION" 2>/dev/null; then
    tmux send-keys -t "${CURRENT_SESSION}:0.0" C-c
fi
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

log "starting_next_run:$NEXT_CFG_REL"
cd "$ROOT"
bash experiments/ops/launch_train_tmux.sh "$NEXT_CFG_REL" "$NEXT_SESSION" 4 experiments.run_supervised_with_eval

log "done"
