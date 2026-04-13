#!/usr/bin/env bash
set -euo pipefail

ROOT="/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec"
TARGET_DIR="$ROOT/output/Llama31_8b_mntp-supervised/DermL2V/SA_Fusion/SlerpMixCSE_k16_SAFusionRouter_aux0p1_onormrms"
CFG="$ROOT/train_configs/StructuredSelfAttnFusion/Llama31-8b-inst-mntp-supervised_SAFusionRouter@DermVariantsSFT.json"
RELAUNCH_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 /home/gbw_21307130160/anaconda3/envs/llm2vec/bin/torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval_v5 $CFG"

echo "waiting_for_checkpoint_80:$TARGET_DIR"
while ! find "$TARGET_DIR" -path "*/checkpoint-80/trainer_state.json" | grep -q checkpoint-80; do
    sleep 30
done

echo "checkpoint_80_detected"
tmux send-keys -t TR:0 C-c
sleep 10

while ps -eo cmd | grep -E "experiments.run_supervised_with_eval_v5|torchrun --standalone --nproc_per_node=4" | grep -v grep >/dev/null; do
    sleep 10
done

echo "restarting_with_aux0p01"
tmux send-keys -t TR:0 "$RELAUNCH_CMD" C-m
echo "done"
