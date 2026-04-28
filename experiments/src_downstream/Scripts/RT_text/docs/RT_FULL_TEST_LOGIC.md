# RT-full Test Logic

## Pipeline

1. `launch/sweep_checkpoints.py` reads an eval config from `configs/`.
2. For each configured checkpoint, it calls `launch/run_full_checkpoint.py`.
3. `run_full_checkpoint.py` resolves the checkpoint through `lib/checkpoint_common.py`, builds the output directory, then launches:

```bash
python -m experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_l2v_multi_full
```

4. `nonhomo_RT_l2v_multi_full.py` loads the model once per checkpoint and evaluates each configured RT dataset.
5. Each checkpoint writes flat dataset result files under:

```text
<rt_nonhomo_full_output_root>/<model family>/<method>/<params>/cp<step>/<dataset>.json
```

6. `summary/render_method_sweep_summary_at10.py` renders one checkpoint-sweep markdown table.
7. `summary/render_family_best_summary.py` renders one family-level best-checkpoint markdown table.

## Metrics

Each dataset result JSON stores retrieval metrics as 0-1 fractions:

- `NDCG@3`, `NDCG@5`, `NDCG@10`
- `Recall@3`, `Recall@5`, `Recall@10`

Markdown summaries report these values as percentages. The sweep summary computes:

```text
Avg_NDCG@10 = mean(NDCG@10 over configured datasets)
Avg_Recall@10 = mean(Recall@10 over configured datasets)
Avg = (Avg_NDCG@10 + Avg_Recall@10) / 2
```

## Current Loss02 Run

Config:

```text
experiments/src_downstream/Scripts/RT_text/configs/derml2v_loss02_rt_full_eval_paths.json
```

Default RT-full encode batch size for this config:

```text
192
```

Training run root:

```text
output/Llama31_8b_mntp-supervised/DermL2VLoss02/SM/lr2e-5_k16_lerp/DermVariants_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-2048_l-512_bidirectional-True_e-2_s-42_w-10_lr-2e-05_lora_r-16
```

Output layout:

```text
output/downstream/DermL2V/RT_text/nonhomo_full/DermL2VLoss02/SM/k16_lerp_lr2e-5/cp<step>/<dataset>.json
```

## Local Dataset Set

The RT-full config now uses the four canonical datasets under `local_info/rt_full_data/`:

- `DermSynth_knowledgebase`
- `MedMCQA_RT`
- `MedQuAD_dermatology_qa_retrieval_doclt300`
- `sce_retrieval`

All new RT-full evaluations should point to these local canonical files instead of older benchmark copies or machine-specific `/storage/...` paths.
