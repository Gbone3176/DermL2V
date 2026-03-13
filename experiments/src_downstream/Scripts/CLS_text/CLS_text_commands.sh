
# bert
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_bert experiments/src_downstream/Scripts/CLS_text/CLS_text-BERT.json

#gpt-2
CUDA_VISIBLE_DEVICES=3 python -m BLURB-src.seqcls.run_seqcls_gpt experiments/src_downstream/Scripts/CLS_text/CLS_text-gpt2.json

#Qwen3 Embedding
CUDA_VISIBLE_DEVICES=5 python -m BLURB-src.seqcls.run_seqcls_qwen_emb experiments/src_downstream/Scripts/CLS_text/CLS_text-qwen3_emb.json

#LLM2Vec:basemodel
CUDA_VISIBLE_DEVICES=2 python -m BLURB-src.seqcls.run_seqcls_llm2vec experiments/src_downstream/Scripts/CLS_text/CLS_text-llm2vec-Llama31_8B_Inst.json

CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-llm2vec-Llama31_8B_Inst.json

#LLM2Vec:Derml2v
# best_weight_path
# Baseline_Datav2: /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2304_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-270
# Baseline_MixCSE_Datav2: /storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_MixCSE_DermData2/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-1536_l-512_bidirectional-True_e-5_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-140
CUDA_VISIBLE_DEVICES=3 python -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-Derml2v.json

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-Derml2v.json



#LLM2Vec:Derml2v_resattn
CUDA_VISIBLE_DEVICES=4 python -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-Derml2v_resattn.json

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-Derml2v_resattn.json