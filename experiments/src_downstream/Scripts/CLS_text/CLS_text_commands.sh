
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
CUDA_VISIBLE_DEVICES=4 python -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-Derml2v.json

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-Derml2v.json

#LLM2Vec:Derml2v_resattn
CUDA_VISIBLE_DEVICES=4 python -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-Derml2v_resattn.json

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 -m BLURB-src.seqcls.run_seqcls_llm2vec_inst experiments/src_downstream/Scripts/CLS_text/CLS_text-Derml2v_resattn.json