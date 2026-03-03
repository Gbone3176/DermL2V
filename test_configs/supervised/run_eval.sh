#!/bin/bash

# 切换到项目根目录
cd /storage/BioMedNLP/llm2vec

CUDA_VISIBLE_DEVICES=5 python -m experiments.test_supervised "test_configs/supervised/test_llama_3_2_1b_inst_cc.json"

# CUDA_VISIBLE_DEVICES=5 python -m experiments.test_supervised "test_configs/supervised/test_sheared_llama_1p3b_mntp_sup.json"

# CUDA_VISIBLE_DEVICES=5 python -m experiments.test_supervised "test_configs/supervised/test_llama_32_3b_inst_mntp_sup.json"