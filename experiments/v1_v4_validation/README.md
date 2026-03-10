# V1 vs V4 Validation Retrieval

This directory contains a small validation-time retrieval benchmark for
comparing `llm2vecV1` and `llm2vecV4` on the same dataset split.

The script evaluates three metrics:

- `recall@10`
- `ndcg@10`
- `avg_recall_ndcg@10` = `(recall@10 + ndcg@10) / 2`

## Recommended command

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/v1_v4_validation/run_v1_v4_validation.py \
  train_configs/Baseline_MixCSE_Fusion/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT_mixcse_fusion.json \
  --split validation \
  --batch-size 16
```

## Notes

- The script reads model and dataset paths from the training config JSON.
- By default it overrides the config pooling mode to `mean` for both V1 and V4.
- V4 enables `cross_layer_lf_prepend` and evaluates the same checkpoint as V1.
- Results are written under `experiments/v1_v4_validation/results/`.
