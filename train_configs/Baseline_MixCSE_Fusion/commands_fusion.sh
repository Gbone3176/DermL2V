###########
#  train  #
###########


# Baseline + MixCSE + Fusion
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_supervised_fusion_withEval train_configs/Baseline_MixCSE_Fusion/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT_mixcse_fusion.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised_fusion_withEval train_configs/Baseline_MixCSE_Fusion/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT_mixcse_fusion.json

# Resume training
CUDA_VISIBLE_DEVICES=0,1 bash experiments/resume_mixcse_fusion.sh


###########
#  test  #
###########

# supervised training
CUDA_VISIBLE_DEVICES=0 python -m experiments.test_supervised test_configs/supervised/test-mntp-simcse-supervised-Llama31-8b_DermVariantsSFT.json



######################################
# task2: SkinCAP Disease Classification
######################################

CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/task2/task2-uni-Llama3.1Inst.json
