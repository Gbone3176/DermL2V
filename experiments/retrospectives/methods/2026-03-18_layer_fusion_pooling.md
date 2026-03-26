# Layer Fusion Pooling (LLM2VecV3)

## Context
- Config: `train_configs/Baseline_MixCSE_Fusion/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT_mixcse_fusion.json`
- Runner: `experiments/run_supervised_fusion_withEval.py`
- Design notes: `train_configs/Baseline_MixCSE_Fusion/FUSION_MODULE_NOTES.md`

## Method Intent
- New `pooling_mode="layer_fusion"` learns weighted combination of last K layers.
- Adds router/norm/gamma parameters to improve representation quality over single-layer pooling.

## Plausible Failure Reasons
- If backbone is mostly frozen and fusion head is small, capacity may be insufficient to fix dataset-level inconsistency.
- If training both LoRA and fusion, optimization targets can compete.
- Lower LR and changed schedule may confound attribution of gains/failures to fusion itself.

## Decision
- Keep as architecture branch, but compare against mean pooling under matched optimization budget.

## Next Checks
- [ ] Fusion-only vs LoRA+fusion controlled comparison.
- [ ] Log learned layer weights across steps to detect collapse to one layer.
