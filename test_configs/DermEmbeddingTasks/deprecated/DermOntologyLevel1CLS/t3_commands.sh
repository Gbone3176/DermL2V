# bert
CUDA_VISIBLE_DEVICES=4 python -m BLURB-src.seqcls.run_seqcls_bert test_configs/DermEmbeddingTasks/task3/task3-BERT.json
# gpt2
CUDA_VISIBLE_DEVICES=1 python -m BLURB-src.seqcls.run_seqcls_gpt2 test_configs/DermEmbeddingTasks/task3/task3-gpt2.json
# LLM2Vec
CUDA_VISIBLE_DEVICES=4 python -m BLURB-src.seqcls.run_seqcls_llm2vec test_configs/DermEmbeddingTasks/task3/task3-llm2vec-Llama31_8B_Inst.json

CUDA_VISIBLE_DEVICES=7 python -m BLURB-src.seqcls.run_seqcls_llm2vec test_configs/DermEmbeddingTasks/task3/task3-llm2vec-Llama31_8B_Inst-Derm1M_10QA_MixCSE.json