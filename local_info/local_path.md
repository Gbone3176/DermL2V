# Local Paths
This file records frequently used local paths for this workspace. These paths are machine-specific and should not be committed to git.

## Repo
- repo_root: `/mnt/nas1/disk06/bowenguo/codes/DermL2V`

## Base Models and Adapters

- llama31_8b_instruct: `/mnt/nas1/disk06/bowenguo/cache/modelscope_cache/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
- llama31_8b_instruct_mntp: `/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291`
- llama31_8b_instruct_mntp: `/mnt/nas1/disk06/bowenguo/cache/huggingface_cache/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db`

- qwen3-embedding-8B: `/mnt/nas1/disk06/bowenguo/cache/modelscope_cache/hub/models/Qwen/Qwen3-Embedding-8B`


## Datasets
### train
- dermvariants_data: `/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermVariantsData`

### test
- RT:
    - `/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/eval3-text-benchmark_split_choices.jsonl`
    - `/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/MedMCQA_RT_query_doc.jsonl`
    - `/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/Text_RT/MedQuAD_dermatology_qa_retrieval.jsonl`

- vis_mcs_data:`/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/exp4-VisualMatching/VisualizationVariations_task.jsonl`

-CLS-skincap: `/mnt/nas1/disk06/bowenguo/datasets/image-text/Derm1M/DermEmbeddingBenchmark/skincap-DiseaseClassification`


## Training Output Roots

- llama31_8b_output_root: `/mnt/nas1/disk06/bowenguo/codes/DermL2V/output/Llama31_8b_mntp-supervised`
- dermvariants_output_root: `/mnt/nas1/disk06/bowenguo/codes/DermL2V/output/Llama31_8b_mntp-supervised/DermVariants`
- derml2v_output_root: `/mnt/nas1/disk06/bowenguo/codes/DermL2V/output/Llama31_8b_mntp-supervised/DermL2V`


## Notes

- Most nonhomo RT sweep scripts expect a training run root and then append `checkpoint-<step>`.
- `cp_0` in those scripts usually means base model + shared adapters only, not the fine-tuned checkpoint directory above.
- Prefer adding new machine-specific paths here instead of hardcoding them repeatedly in scripts.
