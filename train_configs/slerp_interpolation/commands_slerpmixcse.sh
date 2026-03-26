###########
#  train  #
###########


# SlerpMixCSE with HardNegativeNLLLossV4
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_supervised_with_eval train_configs/slerp_interpolation/Llama31-8b-inst-mntp-supervised_SlerpMixCSE@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/slerp_interpolation/Llama31-8b-inst-mntp-supervised_SlerpMixCSE@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/slerp_interpolation/Llama31-8b-inst-mntp-supervised_SlerpMixCSE@DermVariantsSFT.json

