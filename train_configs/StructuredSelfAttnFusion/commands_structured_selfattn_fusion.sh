###########
#  train  #
###########

PYTHON_BIN="/home/gbw_21307130160/anaconda3/envs/llm2vec/bin/python"
TORCHRUN_BIN="/home/gbw_21307130160/anaconda3/envs/llm2vec/bin/torchrun"

# Structured self-attention fusion pooling + HardNegativeNLLLossV6
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m experiments.run_supervised_with_eval_v5 \
    train_configs/StructuredSelfAttnFusion/Llama31-8b-inst-mntp-supervised_SAFusionRouter@DermVariantsSFT_debug.json

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -m experiments.run_supervised_with_eval_v5 \
    train_configs/StructuredSelfAttnFusion/Llama31-8b-inst-mntp-supervised_SAFusionRouter@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
    -m experiments.run_supervised_with_eval_v5 \
    train_configs/StructuredSelfAttnFusion/Llama31-8b-inst-mntp-supervised_SAFusionRouter@DermVariantsSFT.json
