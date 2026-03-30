###########
#  train  #
###########


# Baseline setting
# original LLM2Vec architecture + base model
# + merged mntp weights
# + merged mntp-supervised weights
# data: keep current DermVariants instruction/query-positive-negative format
# loss: HardNegativeNLLLossV0

# 8B baseline, single GPU
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval train_configs/baseline/Llama31-8b-inst-mntp-supervised@DermVariantsSFT.json

# 8B baseline debug, single GPU
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval train_configs/baseline/Llama31-8b-inst-mntp-supervised@DermVariantsSFT_debug.json

# 8B baseline, 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/baseline/Llama31-8b-inst-mntp-supervised@DermVariantsSFT.json

# 8B baseline, 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/baseline/Llama31-8b-inst-mntp-supervised@DermVariantsSFT.json

# 1.3B baseline, single GPU
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval train_configs/baseline/MetaLlama32_1p3B_inst-mntp_supervised@DermVariantsSFT.json
