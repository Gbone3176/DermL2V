###########
#  train  #
###########


# Structured self-attention pooling + top-k shared SlerpMixCSE + AnglE-inspired loss
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_supervised_with_eval train_configs/SM_AngleLoss/Llama31-8b-inst-mntp-supervised_StructuredSelfAttn_k128_gamma0p001_aux0p001_cw1p0_aw0p02@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/SM_AngleLoss/Llama31-8b-inst-mntp-supervised_StructuredSelfAttn_k128_gamma0p001_aux0p001_cw1p0_aw0p02@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/SM_AngleLoss/Llama31-8b-inst-mntp-supervised_StructuredSelfAttn_k128_gamma0p001_aux0p001_cw1p0_aw0p02@DermVariantsSFT.json
