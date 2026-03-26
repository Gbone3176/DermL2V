
# bert
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_bert test_configs/DermEmbeddingTasks/CLS_skincap-DiseaseClassification/task2-BERT.json

#gpt-2
CUDA_VISIBLE_DEVICES=2 python -m BLURB-src.seqcls.run_seqcls_gpt test_configs/DermEmbeddingTasks/CLS_skincap-DiseaseClassification/task2-gpt2.json

#Qwen3 Embedding
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_qwen_emb test_configs/DermEmbeddingTasks/CLS_skincap-DiseaseClassification/task2-qwen3_emb.json

#LLM2Vec:basemodel
CUDA_VISIBLE_DEVICES=2 python -m BLURB-src.seqcls.run_seqcls_llm2vec test_configs/DermEmbeddingTasks/CLS_skincap-DiseaseClassification/task2-llm2vec-Llama31_8B_Inst.json

#LLM2Vec:Derm1M_10QA_MixCSE
CUDA_VISIBLE_DEVICES=5 python -m BLURB-src.seqcls.run_seqcls_llm2vec test_configs/DermEmbeddingTasks/CLS_skincap-DiseaseClassification/task2-llm2vec-Llama31_8B_Inst-Derm1M_10QA_MixCSE.json