# Loss Versions Retrospective (V0 to V5)

## Context
- Registry: `llm2vec/loss/utils.py`
- Implementations:
  - `HardNegativeNLLLossV0.py`
  - `HardNegativeNLLLossV1.py`
  - `HardNegativeNLLLossV2.py`
  - `HardNegativeNLLLossV3.py`
  - `HardNegativeNLLLossV4.py`
  - `HardNegativeNLLLossV5.py`

## Version Summary
- V0: basic in-batch + explicit negatives CE.
- V1: MixCSE-style fixed-lambda mixed hard negative.
- V2: V1 + focal reweighting (`gamma`).
- V3: dynamic lambda + margin-aware mixed-negative penalty.
- V4: fixed-lambda Slerp/Lerp mixed negative.
- V5: V4 + top-k shared mixed negatives across batch.

## Why Many Variants Still May Fail
- Most variants still depend on hardest-negative mining quality; if mined negatives are noisy, all versions inherit instability.
- More complex losses introduce more hyperparameters, raising tuning cost and variance under fixed compute budget.
- If subset objectives conflict (already observed in data analysis), loss refinement alone cannot fix label-space inconsistency.

## Decision
- Keep the loss family, but stop full-run hyperparameter sweeps without low-cost screening.

## Next Checks
- [ ] Build a small-step benchmark (e.g., first 300-500 steps) for loss ranking before full 30h runs.
- [ ] Log per-subset margin statistics under each loss version.
- [ ] Freeze one stable baseline for regression testing.
