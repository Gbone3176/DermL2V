###########
#  train  #
###########


# Structured self-attention pooling + HardNegativeNLLLossV6
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_supervised_with_eval train_configs/StructuredSelfAttnPooling/Llama31-8b-inst-mntp-supervised_StructuredSelfAttn@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/StructuredSelfAttnPooling/Llama31-8b-inst-mntp-supervised_StructuredSelfAttn@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/StructuredSelfAttnPooling/Llama31-8b-inst-mntp-supervised_StructuredSelfAttn@DermVariantsSFT.json


# Structured self-attention pooling ablation:
# original contrastive loss + structured self-attention auxiliary regularization
CUDA_VISIBLE_DEVICES=7 python -m experiments.run_supervised_with_eval train_configs/StructuredSelfAttnPooling/Llama31-8b-inst_woSlerpMixCSE_StructuredSelfAttn@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/StructuredSelfAttnPooling/Llama31-8b-inst_woSlerpMixCSE_StructuredSelfAttn@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/StructuredSelfAttnPooling/Llama31-8b-inst_woSlerpMixCSE_StructuredSelfAttn@DermVariantsSFT.json
