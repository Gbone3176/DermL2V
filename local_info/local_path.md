# Local Paths
This file records frequently used local paths for this workspace. These paths are machine-specific and should not be committed to git.

## Repo
- repo_root: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec`

## Base Models and Adapters

- llama31_8b_instruct: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/cache/modelscope/hub/model/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- llama31_8b_instruct_mntp: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/cache/huggingface/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291`
- llama31_8b_instruct_mntp_supervised: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/cache/huggingface/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db`

- qwen3-embedding-0.6B: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/cache/modelscope/models/Qwen/Qwen3-Embedding-0.6B` 
- qwen3-embedding-8B: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/cache/modelscope/models/Qwen/Qwen3-Embedding-8B`


## Datasets
### train
- dermvariants_data: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/text-img/dermatoscop/DermVariantsData`


## Training Output Roots

- llama31_8b_output_root: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised`
- dermvariants_output_root: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec/Llama31_8b_mntp-supervised/DermVariants`
- derml2v_baseline_output_root: `/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/storage/BioMedNLP/llm2vec/Llama31_8b_mntp-supervised/DermL2V/baseline`



## Notes

- Most nonhomo RT sweep scripts expect a training run root and then append `checkpoint-<step>`.
- `cp_0` in those scripts usually means base model + shared adapters only, not the fine-tuned checkpoint directory above.
- Prefer adding new machine-specific paths here instead of hardcoding them repeatedly in scripts.
