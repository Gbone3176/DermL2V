# FocalMixCSE (HardNegativeNLLLossV3)

## Context
- Configs:
  - `train_configs/FocalMixCSE/Llama31-8b-inst-mntp-supervised_FocalMixCSE@DermVariantsSFT.json`
  - `train_configs/FocalMixCSE/Llama31-8b-inst-mntp-supervised_MixCSE@DermVariantsSFT.json`
- Runner: `experiments/run_supervised_FocalMixCSE.py`

## Method Intent
- Use dynamic lambda and margin-aware mixed-negative penalty to focus on harder samples.
- In focal variant, `mix_weight=1.0`; in baseline-under-V3 config, `mix_weight=0.0` as comparison.

## Plausible Failure Reasons
- Dynamic lambda is driven by current hard-vs-pos scores; if negatives are noisy, control signal becomes noisy.
- Margin and focal effects can over-amplify already ambiguous samples (especially boundary/noisy negatives).
- Higher method complexity can mask true bottleneck from data composition.

## Decision
- Use this method only after validating negative quality per subset.

## Next Checks
- [ ] Plot distribution of `s_pos - s_hard` by subset during training.
- [ ] Compare V3 with `mix_weight=0` vs `1` under identical seed and short-run budget.
