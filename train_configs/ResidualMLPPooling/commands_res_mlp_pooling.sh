###########
#  train  #
###########


# Residual MLP Pooling + HardNegativeNLLLossV5
# shared slerp mixed negatives:
#   lam=0.2
#   interpolation_mode=slerp
#   shared_mix_topk=16

# Debug
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval_v4 train_configs/ResidualMLPPooling/Llama31-8b-inst-mntp-supervised_ResMLPPooling@DermVariantsSFT_debug.json

# Full training, 1 GPU
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval_v4 train_configs/ResidualMLPPooling/Llama31-8b-inst-mntp-supervised_ResMLPPooling@DermVariantsSFT.json

# Full training, 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval_v4 train_configs/ResidualMLPPooling/Llama31-8b-inst-mntp-supervised_ResMLPPooling@DermVariantsSFT.json

# Full training, 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval_v4 train_configs/ResidualMLPPooling/Llama31-8b-inst-mntp-supervised_ResMLPPooling@DermVariantsSFT.json
