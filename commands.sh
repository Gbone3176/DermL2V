###########
#  train  #
###########

# mntp
CUDA_VISIBLE_DEVICES=2,3,5,6 torchrun --standalone --nproc_per_node=4 -m experiments.run_mntp_v0 train_configs/mntp/MetaLlama3.1_Derm1M.json

# simcse
CUDA_VISIBLE_DEVICES=2 python -m experiments.run_simcse_v0 train_configs/simcse/MetaLlama3.1_debug.json

# Supervised
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_supervised train_configs/supervised/Llama31-8b-inst-mntp-simcse@supervisedL2V.json

CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised train_configs/supervised/Llama31-8b-inst-mntp-simcse@supervisedL2V.json

CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval train_configs/supervised/Llama31-8b-inst-mntp-simcse@supervisedwithevalV2-L2V.json

## Eval - Supervised
CUDA_VISIBLE_DEVICES=0 python -m experiments.test_supervised train_configs/supervised/test-mntp-simcse-supervised-Llama31-8b-train.json



################################################
# exp1: Skin Visual Description Classification
################################################
# L1: Bi-model
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp1/exp1-Bi-MetaLlama3.1_inst_L2V_Derm1M.json

# L2: Bi-mntp-simcse-model
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp1/exp1-Bi-mntp-simcse-MetaLlama3.1_inst_L2V_Derm1M.json

# L3: Bi-mntp-supervised-model
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp1/exp1-Bi-mntp-supervised-MetaLlama3.1_inst_L2V_Derm1M.json

# L4: Bi-mntp-supervised-DermSFT-model
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp1/exp1-DermSFT-MetaLlama3.1_inst_L2V_Derm1M.json

# BERT
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls train_configs/DermEmbeddingTasks/exp1/exp1-BioLinkBERT_base.json


######################################
# exp2: SkinCAP Disease Classification
######################################

# L3: Bi-mntp-supervised-model
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp2/exp2-mntp-supervised-MetaLlama3.1_inst_L2V_Derm1M.json

# L4: Bi-mntp-supervised-DermSFT-model
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp2/exp2-mntp-supervised-SFT-MetaLlama3.1_inst_L2V_Derm1M.json



######################################
# exp3: Ontology Level1 Classification
######################################

# L3: Bi-mntp-supervised-model
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp3/exp3-mntp-supervised-MetaLlama3.1_inst_L2V_Derm1M.json

# L4: Bi-mntp-supervised-DermSFT-model
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp3/exp3-mntp-supervised-SFT-MetaLlama3.1_inst_L2V_Derm1M.json


######################################
# exp4: DermQA retrieval
######################################

# exp4-zeroshot: Llama3.1-8B-inst-official
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_DermQA_zeroshot_v2 train_configs/DermEmbeddingTasks/exp4/exp4-V2-mntp-simcse-supervised-Llama31-8b-official.json

# exp4-zeroshot:  Llama3.1-8B-inst-train
CUDA_VISIBLE_DEVICES=1 python -m experiments.run_DermQA_zeroshot_v2 train_configs/DermEmbeddingTasks/exp4/exp4-V2-mntp-simcse-supervised-Llama31-8b-train.json

# exp4-zeroshot: LLM2CLIP-basemodel
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_DermQA_zeroshot_v2 train_configs/DermEmbeddingTasks/exp4/exp4-V2-LLM2CLIP-Llama3.2-1B-CC-inst.json

# exp4-zeroshot: bert
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_DermQA_zeroshot_v2_bert train_configs/DermEmbeddingTasks/exp4/exp4-DermQA_V2_bert.json

# exp4-zeroshot: bert-inst
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_DermQA_zeroshot_v2_bert_inst train_configs/DermEmbeddingTasks/exp4/exp4-DermQA_V2_bert.json



######################################
# exp5: SimSenRetriieval
######################################

# exp5-LLM2Vec: 
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_DermSimRetrieval_l2v train_configs/DermEmbeddingTasks/exp5/exp5-uni-zeroshot-MetaLlama3.1_inst_L2V_Derm1M.json

# exp5-BERT: 
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_DermSimRetrieval_bert train_configs/DermEmbeddingTasks/exp5/exp5-ClinicalBERT.json