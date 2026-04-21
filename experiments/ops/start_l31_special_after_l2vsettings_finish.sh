#!/usr/bin/env bash
set -euo pipefail

ROOT="/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec"

CURRENT_CFG_REL="train_configs/Llama31_8B_workflow_baseline/baseline/Llama31-8b-inst-mntp-sup@DermVariantsSFT-LLM2vecSettings.json"
NEXT_CFG_REL="train_configs/Llama31_8B_workflow_baseline/baseline/Llama31-8b-inst-mntp-sup@DermVariantsSFT-special.json"
NEXT_SESSION="l31_base_special"
NPROC=4
MODULE_NAME="experiments.run_supervised_with_eval"

log() {
    echo "[$(date '+%F %T')] $*"
}

current_pids() {
    pgrep -f "$CURRENT_CFG_REL" || true
}

log "waiting_for_run_exit:$CURRENT_CFG_REL"
while [[ -n "$(current_pids)" ]]; do
    sleep 30
done

log "current_run_finished"

if tmux has-session -t "$NEXT_SESSION" 2>/dev/null; then
    log "next_session_already_exists:$NEXT_SESSION"
    exit 0
fi

cd "$ROOT"
log "starting_next_run:$NEXT_CFG_REL"
bash experiments/ops/launch_train_tmux.sh "$NEXT_CFG_REL" "$NEXT_SESSION" "$NPROC" "$MODULE_NAME"

log "done"
