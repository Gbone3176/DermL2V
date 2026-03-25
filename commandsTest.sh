# supervised training 单次测试 
CUDA_VISIBLE_DEVICES=0 python -m experiments.test_supervised test_configs/supervised/test-mntp-simcse-supervised-Llama31-8b_DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0 python -m experiments.test_supervised test_configs/supervised/test-mntp-simcse-supervised-Llama31-8b_DermVariantsSFT.json

CUDA_VISIBLE_DEVICES=0 python -m experiments.test_qwen3_embedding test_configs/supervised/test-mntp-simcse-supervised-Qwen3_8B_embedding.json


# 一次性测试所有的checkpoints

cd /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec/scripts