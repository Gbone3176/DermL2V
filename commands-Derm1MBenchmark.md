# mntp
## Meta-Llama-3.1-8B-Instruct
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_mntp train_configs/mntp/MetaLlama3.1_inst_L2V_Derm1M.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_mntp_multiGPU train_configs/mntp/MetaLlama3.1_inst_L2V_Derm1M.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_simcse train_configs/mntp/MetaLlama3.1_inst_L2V_Derm1M.json


``` 

# simcse
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_simcse train_configs/simcse/mntp-MetaLlama3.1_inst_L2V_Derm1M.json

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m experiments.run_simcse train_configs/simcse/mntp-MetaLlama3.1_inst_L2V_Derm1M.json
``` 

# supervised
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_supervised_with_eval train_configs/supervised/DermQA-Llama31-8b-inst-mntp-simcse@supervisedwithevalV2-L2V.json
```

# contrastive experiments
# downstream task



## Classification_task_1: Sentence Classification

```bash
##uni
CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp2/exp2-uni-MetaLlama3.1_inst_L2V_Derm1M.json

CUDA_VISIBLE_DEVICES=0 python -m BLURB-src.seqcls.run_seqcls_llm2vec train_configs/DermEmbeddingTasks/exp2/exp2-mntp-simcse-Llama3.1Inst.json
```


## Retrieval_task_2: Text-Textetrieval

```bash

```
