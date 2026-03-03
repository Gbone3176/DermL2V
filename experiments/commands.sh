###########
#  train  #
###########

# Supervised
CUDA_VISIBLE_DEVICES=7 python -m experiments.run_supervised_withEval train_configs/supervised/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=7 python -m experiments.run_supervised_withEval train_configs/supervised/MetaLlama32_1p3B_inst-mntp_supervised@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_withEval train_configs/supervised/MetaLlama32_1p3B_inst-mntp_supervised@DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=3,5,6,7 torchrun --standalone --nproc_per_node=4 -m experiments.run_supervised_withEval train_configs/supervised/MetaLlama32_1p3B_inst-mntp_supervised@DermVariantsSFT.json

###########
#  test  #
###########

# supervised training
CUDA_VISIBLE_DEVICES=0 python -m experiments.test_supervised test_configs/supervised/test-mntp-simcse-supervised-Llama31-8b_DermVariantsSFT.json



######################################
# task2: SkinCAP Disease Classification
######################################

CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/task2/task2-uni-Llama3.1Inst.json





# 以下命令均弃用

# ######################################
# # task3: Ontology Level1 Classification
# ######################################
# CUDA_VISIBLE_DEVICES=1 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/task3/task3-uni-Llama3.1Inst.json


# ######################################
# # task4: Visual Error Discimination
# ######################################
# CUDA_VISIBLE_DEVICES=1 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/task4/task4-uni-Llama3.1Inst.json



# ######################################
# # task5: SimSenRetriieval
# ######################################

# # task5-LLM2Vec: 
# CUDA_VISIBLE_DEVICES=2 python -m experiments.run_DermSimRetrieval train_configs/DermEmbeddingTasks/task5/task5-uni-zeroshot-MetaLlama3.1_inst_L2V_Derm1M.json

# # task5-BERT: 
# CUDA_VISIBLE_DEVICES=2 python -m experiments.run_DermSimRetrieval_bert train_configs/DermEmbeddingTasks/task5/task5-ClinicalBERT.json

