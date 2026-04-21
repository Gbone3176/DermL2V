###########
#  train  #
###########


# 1.3B baseline workflow
# base model: the previously runnable 1.3B setup referenced by
# train_configs/Archive/supervised/MetaLlama32_1p3B_inst-mntp-sup@DermVariantsSFT.json
# params aligned with current 8B baseline workflow

# 1.3B baseline, single GPU
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval train_configs/Llama32_1p3B_workflow_baseline/baseline/MetaLlama32_1p3B_inst-mntp-sup@DermVariantsSFT.json

# 1.3B baseline, 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_with_eval train_configs/Llama32_1p3B_workflow_baseline/baseline/MetaLlama32_1p3B_inst-mntp-sup@DermVariantsSFT.json

# 1.3B baseline, 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/Llama32_1p3B_workflow_baseline/baseline/MetaLlama32_1p3B_inst-mntp-sup@DermVariantsSFT.json

# 1.3B SA baseline (earliest residual injection), 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/Llama32_1p3B_workflow_baseline/SA/MetaLlama32_1p3B_inst_SA@DermVariantsSFT.json

# 1.3B SM k128, 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/Llama32_1p3B_workflow_baseline/SM/MetaLlama32_1p3B_inst-mntp-sup_SM_k128@DermVariantsSFT.json

# 1.3B SM+SA k128, 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_with_eval train_configs/Llama32_1p3B_workflow_baseline/SA_SM/MetaLlama32_1p3B_inst-mntp-sup_SM_SA_k128_gamma0p001_aux0p001@DermVariantsSFT.json
