# SlerpMixCSE (HardNegativeNLLLossV4)

## Context
- Config: `train_configs/slerp_interpolation/Llama31-8b-inst-mntp-supervised_SlerpMixCSE@DermVariantsSFT.json`
- Runner: `experiments/run_supervised_with_eval.py`
- Command: `train_configs/slerp_interpolation/commands_slerpmixcse.sh`

## Method Intent
- Replace linear interpolation with spherical interpolation between positive and hardest negative.
- Keep fixed lambda (`lam=0.2`) and append one row-specific mixed negative.

## Plausible Failure Reasons
- Geometry refinement (Slerp) helps only if mined hard negatives are informative; it cannot correct mislabeled or over-hard negatives.
- Single mixed negative per row may have limited effect under large candidate pool.
- Fixed lambda may be suboptimal for heterogeneous subsets with very different difficulty structures.

## Decision
- Keep as controlled variant, not default training path.

## Next Checks
- [ ] Compare Lerp vs Slerp under same config except interpolation mode.
- [ ] Evaluate subset-wise gains to see whether Slerp only helps a minority subset.
