###########
#  train  #
###########

# mntp
CUDA_VISIBLE_DEVICES=2,3,5,6 torchrun --standalone --nproc_per_node=4 -m experiments.run_mntp_v0 train_configs/mntp/MetaLlama3.1_Derm1M.json

# simcse
CUDA_VISIBLE_DEVICES=2 python -m experiments.run_simcse_v0 train_configs/simcse/MetaLlama3.1_debug.json

# Supervised

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval_V0 train_configs/supervised/Llama31-8b-inst-mntp-supervised@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval train_configs/supervised/Llama31-8b-inst-mntp-supervised@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/supervised/Llama31-8b-inst-mntp-supervised@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/supervised/Llama31-8b-inst-mntp-supervised@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/supervised/Llama31-8b-inst-mntp-supervised@DermVariantsSFT_debug.json
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/supervised/Llama31-8b-inst-mntp-supervised@DermVariantsSFT_debug.json


# McGill-NLP/LLM2Vec-Sheared-LLaMA
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval train_configs/supervised/MetaLlama32_1p3B_inst-mntp_supervised@DermVariantsSFT.json

## Eval - Supervised
CUDA_VISIBLE_DEVICES=0 python -m experiments.test_supervised train_configs/supervised/test-mntp-simcse-supervised-Llama31-8b-train.json
