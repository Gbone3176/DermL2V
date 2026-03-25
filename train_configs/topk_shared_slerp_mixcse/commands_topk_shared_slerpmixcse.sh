###########
#  train  #
###########


# Top-k shared SlerpMixCSE with HardNegativeNLLLossV5
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_supervised_with_eval train_configs/topk_shared_slerp_mixcse/Llama31-8b-inst-mntp-supervised_TopKSharedSlerpMixCSE@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/topk_shared_slerp_mixcse/Llama31-8b-inst-mntp-supervised_TopKSharedSlerpMixCSE@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/topk_shared_slerp_mixcse/Llama31-8b-inst-mntp-supervised_TopKSharedSlerpMixCSE@DermVariantsSFT.json
