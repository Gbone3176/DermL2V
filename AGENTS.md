# AGENTS.md

## Path Handling

- This repository uses the `local_info/` directory to store frequently used local file paths and machine-specific path references.
- When updating, replacing, or adding file paths in scripts, configs, or commands, check `local_info/` first and prefer those recorded paths over introducing new hardcoded paths.
- If a required path is missing from `local_info/`, add or update the relevant entry there before reusing that path broadly in the project.
- Treat `local_info/` as the default source of truth for local path lookup in this repository.

## Retrieval Evaluation Dataset Policy

- For nonhomogeneous text retrieval evaluation, use `MedQuAD_dermatology_qa_retrieval_doclt300` instead of the original `MedQuAD_dermatology_qa_retrieval`.
- If a script, config, command, or summary currently references the original `MedQuAD_dermatology_qa_retrieval` as an evaluation target, update it to the `doclt300` variant unless the user explicitly asks otherwise.

## Metrics Presentation Policy

- For plots and paper-facing tables in this repository, present retrieval metrics such as NDCG, Recall, and derived averages as percentages rather than 0-1 fractions.
- When adapting or creating plotting/table-generation scripts, convert metric values by multiplying by 100 and label axes/columns explicitly with `%` or `(%)`.
- Apply the same percentage convention consistently across generated markdown summaries when those summaries are intended to support plotting, reporting, or paper table preparation.

## CLS_text Result Sync Policy

- After completing a `CLS_text` experiment whose outputs are written under `output/downstream/DermL2V/CLS_text/` or related classification result folders, update the shared summary file `output/downstream/CLS_text/LP/results.md`.
- When adding a new `CLS_text` result row, record both Macro and Micro sections using the model name from the run directory or config, and compute:
  - `Amacro` = average of `f1_macro`, `mAP_macro`, and `roc_auc_macro`
  - `Amicro` = average of `f1_micro`, `mAP_micro`, and `roc_auc_micro`
- Match the existing numeric style in `output/downstream/CLS_text/LP/results.md` unless the user explicitly asks to reformat the whole table.

## Project Memory

- In this repository, when the user says `项目记忆`, treat it as referring to `AGENTS.md`.
- If the user asks to update, persist, or consult `项目记忆`, read or modify `AGENTS.md` unless the user explicitly specifies a different file.

## Sync And Commit Policy

- When the goal is to sync code to a training machine, only commit and push training-related code, configs, and launch scripts.
- Do not include post-processing, plotting, retrospective analysis, result rendering, or monitoring/watch helper scripts in such sync-oriented commits unless the user explicitly asks for them.
- If the worktree also contains unrelated evaluation or post-processing changes, leave them uncommitted by default and mention that they were excluded.

## Ablation Summary Conventions

- For `output/downstream/DermL2V/RT_text/nonhomo_full/ablation.md`, when organizing ablation subsections such as `## SA:gamma`, prefer concise summary tables over expanded metric dumps.
- In these ablation subsection summaries, keep only `@10` retrieval metrics and the global `Avg` unless the user explicitly asks for `@3`, `@5`, or per-dataset metric breakout tables.
- For `## SA:gamma`, a preferred layout is:
  - one `gamma` comparison table with `eval3 NDCG@10`, `eval3 Recall@10`, `MedMCQA NDCG@10`, `MedMCQA Recall@10`, `MedQuAD NDCG@10`, `MedQuAD Recall@10`, and `Avg`
  - if relevant, one auxiliary table for a fixed gamma setting such as `aux sweep at gamma=0.001`, using the same `@10` + `Avg` columns

## RT Evaluation Workflow

- For nonhomogeneous RT retrieval experiments, the default execution flow is two-stage rather than directly running the full three-dataset evaluation.
- Stage 1 is a sweep on `/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl` to evaluate all available checkpoints for a run.
- Use the nonhomo sweep scripts under `experiments/src_downstream/Scripts/RT_text/nonhomo/`, for example:
  - `experiments/src_downstream/Scripts/RT_text/nonhomo/Nonhomo_RT_sweep_8B.sh`
  - `experiments/src_downstream/Scripts/RT_text/nonhomo/Nonhomo_RT_sweep_8B_selfattn.sh`
  Remember that it is needed to run the 2 versions of "woinst" and "inst".
- The sweep stage is the primary early selection step. Always inspect at least:
  - the best-performing step on `eval3-text-benchmark_split_choices`
  - the `step=50` result
- Stage 2 is the full nonhomo RT evaluation on the three RT datasets only after deciding whether to use `step=50` or the best step from Stage 1.
- The three full-eval datasets are:
  - `/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/eval3-text-benchmark_split_choices.jsonl`
  - `/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedMCQA_RT_query_doc.jsonl`
  - `/storage/dataset/dermatoscop/DermEmbeddingBenchmark/RT_text/MedQuAD_dermatology_qa_retrieval_doclt300.jsonl`
- Use the full-eval scripts under `experiments/src_downstream/Scripts/RT_text/nonhomo/full/`, selecting the script that matches the pooling setup. Typical examples:
  - `experiments/src_downstream/Scripts/RT_text/nonhomo/full/nonhomo_RT_derml2v_mean_full.sh`
  - `experiments/src_downstream/Scripts/RT_text/nonhomo/full/nonhomo_RT_derml2v_selfattn_full.sh`
  - `experiments/src_downstream/Scripts/RT_text/nonhomo/full/nonhomo_RT_l2v_selfattn_full.sh`
- When adapting a full-eval script for a new checkpoint, prefer editing or overriding only the model name and checkpoint/adaptor path, while keeping the existing dataset list and output layout unchanged.
- Full nonhomo RT results should be written into `output/downstream/DermL2V/RT_text/nonhomo_full/` using the existing per-dataset JSON layout.
- Keep full nonhomo RT result filenames aligned with the existing model naming style. For DermL2V runs, prefer names like `DermL2V_SM_SA_K128_gamma0p1_aux0p01_cp50.json` and avoid temporary task/script prefixes such as `RT_` when the corresponding run is a DermL2V model result.
- After the full evaluation, update the corresponding summary artifacts, such as:
  - `output/downstream/DermL2V/RT_text/nonhomo_full/ablation.md`
  - `output/downstream/DermL2V/RT_text/collate.ipynb` or its derived summaries when relevant
- When reporting conclusions for a run, explicitly state:
  - which checkpoint was chosen for full evaluation: `step=50` or the sweep-best step
  - how the chosen checkpoint compares with baseline on `eval3-text-benchmark_split_choices`
  - how it performs on the three-dataset full RT evaluation

## DermL2V Ablation Weights

- The current fixed DermL2V ablation checkpoints are:
  - `baseline`: `/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/baseline/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-3_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50`
  - `w/ SA`: `/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA/woSlerpMixCSE_StructuredSelfAttn_aux001_gamma1/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50`
  - `w/ SA, SM`: `/storage/BioMedNLP/llm2vec/output/Llama31_8b_mntp-supervised/DermL2V/SA_SM/SlerpMixCSE_k128_StructuredSelfAttn_gamma0p1_aux0p001/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-structured_selfattn_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16/checkpoint-50`
- When these ablation models are evaluated on RT-homo, use stable model names:
  - `DermL2V_Baseline_cp50`
  - `DermL2V_Baseline_SA_aux001_gamma1_cp50`
  - `DermL2V_Baseline_SM_SA_K128_cp50`

## RT-homo Evaluation Workflow

- RT-homo is no longer treated as a reliable evaluation of model generalization ability in this repository. Do not use homo results as the primary basis for model selection or generalization claims unless the user explicitly asks for homo-only analysis.
- Before starting a new RT-homo run for a DermL2V ablation model, first check whether `output/downstream/RT_text/homo/combined/homo_RT_<model_name>.json` already exists.
- Use `experiments/src_downstream/rt_text/homo/homo_RT_l2v.py` for DermL2V homo evaluation.
- Keep the shared base/adaptor stack unchanged:
  - base model: `/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct`
  - mntp adapter: `/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp/snapshots/34ac7221d7ea81c99f1fc8bc823a167dcb795291`
  - supervised adapter: `/cache/hf_home/hub/models--McGill-NLP--LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised/snapshots/9acedfe23912d2db78e6381cbd388ba7acefc6db`
- Use `pooling_mode="mean"` for `DermL2V_Baseline_cp50`.
- Use `pooling_mode="structured_selfattn"` for `DermL2V_Baseline_SA_aux001_gamma1_cp50` and `DermL2V_Baseline_SM_SA_K128_cp50`.
- Write RT-homo outputs into `output/downstream/RT_text/homo/combined/`.
