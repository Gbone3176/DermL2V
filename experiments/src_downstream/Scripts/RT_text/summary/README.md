# RT Text Summary Renderers

This folder contains small Markdown renderers for RT nonhomo-full result summaries.
The dry-run commands below write to `/tmp` so they do not overwrite official result files.

## `render_method_sweep_summary_at10.py`

Summarizes one DermL2V method/parameter directory across all complete `cp*` checkpoint result folders. It writes a checkpoint-sweep table with per-dataset `NDCG@10`, `Recall@10`, average columns, and bolded best values.

Dry run:

```bash
python experiments/src_downstream/Scripts/RT_text/summary/render_method_sweep_summary_at10.py \
  --method-dir output/downstream/DermL2V/RT_text/nonhomo_full/3p2_1p3b/baseline/lossv0p3 \
  --output /tmp/rt_method_sweep_summary_at10.md
```

## `render_family_best_summary.py`

Summarizes one DermL2V output family from a config file. For each configured method/parameter setting, it selects the best complete `cp*` checkpoint by overall average and renders a family-level best-step table.

Dry run:

```bash
python experiments/src_downstream/Scripts/RT_text/summary/render_family_best_summary.py \
  --config-path experiments/src_downstream/Scripts/RT_text/configs/derml2v_loss02_rt_full_eval_paths.json \
  --output /tmp/rt_family_best_summary.md
```

## `render_global_scoreboard.py`

Builds a global RT nonhomo-full scoreboard for baseline, lexical, biomedical encoder, BME retriever, and LLM-based model result directories. It groups models by family and highlights column-best values.

Dry run:

```bash
python experiments/src_downstream/Scripts/RT_text/summary/render_global_scoreboard.py \
  --root output/downstream/RT_text/nonhomo-full \
  --output /tmp/rt_global_scoreboard.md
```

## `render_contrastive_summary.py`

Summarizes one contrastive or non-DermL2V model result directory into a compact four-dataset table plus overall average.

Dry run:

```bash
python experiments/src_downstream/Scripts/RT_text/summary/render_contrastive_summary.py \
  --model_dir output/downstream/RT_text/nonhomo-full/BioClinicalBERT \
  --output /tmp/rt_contrastive_summary.md
```
