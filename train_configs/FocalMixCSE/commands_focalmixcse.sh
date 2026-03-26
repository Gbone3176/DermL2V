###########
#  train  #
###########


# FocalMixCSE
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_supervised_FocalMixCSE train_configs/FocalMixCSE/Llama31-8b-inst-mntp-supervised_FocalMixCSE@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_FocalMixCSE train_configs/FocalMixCSE/Llama31-8b-inst-mntp-supervised_FocalMixCSE@DermVariantsSFT.json

# MixCSE baseline under HardNegativeNLLLossV3
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_supervised_FocalMixCSE train_configs/FocalMixCSE/Llama31-8b-inst-mntp-supervised_MixCSE@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_FocalMixCSE train_configs/FocalMixCSE/Llama31-8b-inst-mntp-supervised_MixCSE@DermVariantsSFT.json

# Resume training
CUDA_VISIBLE_DEVICES=0,1 bash experiments/resume_focalmixcse.sh

