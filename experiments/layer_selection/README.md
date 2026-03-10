# Layer Selection

This directory contains a frozen-backbone retrieval evaluator for choosing candidate layers before running `LoRA + fusion`.

The script loads the merged backbone defined by a training config:

- `model_name_or_path`
- `peft_model_name_or_path`
- `extra_model_name_or_path`

It then ignores the config's current pooling mode, forces `pooling_mode="mean"`, extracts every hidden layer once, and evaluates:

- each single layer with masked mean pooling
- optional contiguous layer ranges by averaging the selected layer embeddings

This avoids the strongest train/test mismatch from using a separately trained fusion head just to decide which backbone layers are worth fusing.

## Recommended command

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/layer_selection/run_layerwise_pooling_eval.py \
  train_configs/Baseline_MixCSE_Fusion/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT_mixcse_fusion.json \
  --split validation \
  --device cuda \
  --range-mode suffix \
  --window-sizes 4,8,12,16
```

## Useful flags

- `--max-examples 512`
  Use a smaller validation subset for a quick first pass.
- `--range-mode suffix`
  Evaluate only `last k` ranges. This matches the current fusion implementation.
- `--range-mode sliding --window-stride 4`
  Evaluate more general contiguous layer ranges at a higher compute cost.
- `--metric recall_at_1`
  Rank best layers/ranges by a different retrieval metric.

## Output

Results are written to:

`experiments/layer_selection/results/<config_stem>_<split>_layerwise_eval.json`

The JSON includes:

- ranked single-layer metrics
- ranked layer-range metrics
- best layer and best range under the selected primary metric
