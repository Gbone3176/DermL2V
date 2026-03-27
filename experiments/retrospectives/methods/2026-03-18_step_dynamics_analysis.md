# Step Dynamics Analysis

## Context
- Run family: `withEval_QAx10_SlerpMixCSE_DermData2`
- Related config:
  - `train_configs/supervised/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT.json`
- Related trainer state:
  - `output/Llama31_8b_mntp-supervised/DermVariants/withEval_QAx10_SlerpMixCSE_DermData2/.../checkpoint-90/trainer_state.json`
- Analysis script:
  - `experiments/retrospectives/data/analyze_step_dynamics.py`

## Important Correction
- The earlier epoch-boundary explanation applies to the debug config with:
  - `per_device_train_batch_size = 64`
  - `gradient_accumulation_steps = 8`
- This run is different:
  - `per_device_train_batch_size = 16`
  - `gradient_accumulation_steps = 4`
- Therefore:
  - effective samples per optimizer step = `16 * 4 * 4 = 256`
  - effective train size after batching alignment = `136960`
  - steps per epoch = `136960 / 256 = 535`
- So `step 55-85` is not near an epoch boundary for this run.

## What Was Analyzed
- epoch boundary markers
- step-level task composition
- step-level mean `pos_score - neg_score`
- logged `grad_norm`

## Result

### 1. The local dip around step 50-90 is not caused by epoch rollover
- In this run, epoch boundary is at step `535`, not `67`.
- So the local behavior around step `55-85` must come from something else.

### 2. Task composition is fairly stable in the dip window
- Across steps `45-95`, the subset ratios stay in roughly similar ranges:
  - `SemVariants`: about `0.08 - 0.18`
  - `VisVariants`: about `0.13 - 0.27`
  - `DermQA`: about `0.04 - 0.11`
  - `SI1`: about `0.23 - 0.37`
  - `SI2`: about `0.26 - 0.37`
- No abrupt subset swap or dominance flip is visible.

### 3. Mean margin is also relatively stable
- In the same window, step-level mean margin is mostly around:
  - `0.12 - 0.15`
- There is fluctuation, but no strong cliff exactly matching the metric dip.

### 4. `grad_norm` bottoms out early, then slowly rises
- Logged values in the window:
  - step `45`: about `0.918`
  - step `55`: about `0.802`
  - step `65`: about `0.800`
  - step `75`: about `0.840`
  - step `85`: about `0.891`
- This looks like:
  - rapid early stabilization
  - then a low-gradient plateau
  - then mild re-expansion of update magnitude

## Current Interpretation
- The dip is unlikely to be caused by:
  - epoch boundary
  - a sudden shift in subset composition
  - a sudden drop in average hard-negative margin
- It is more likely related to optimization behavior, for example:
  - early easy-fit region followed by harder local adjustments
  - representation geometry changing before downstream retrieval improves again
  - local mismatch between training loss reduction and retrieval validation quality

## Practical Takeaway
- For this run, "data composition shock" is not the main explanation for the `50-90` dip.
- The more likely explanation is:
  - optimization-phase dynamics under mixed-task training
  - combined with noisy or near-boundary hard negatives

## Generated Artifacts
- `experiments/retrospectives/data/step_dynamics_outputs/step_dynamics_full.csv`
- `experiments/retrospectives/data/step_dynamics_outputs/focus_steps_45_95.csv`
- `experiments/retrospectives/data/step_dynamics_outputs/step_dynamics_overview.png`

## Next Actions
- [ ] Compare step-level validation metrics with the same step-level train dynamics in one plot
- [ ] Check whether the dip is driven by one downstream subset rather than the whole validation set
- [ ] Add step-level low-margin ratio, not just mean margin
- [ ] Compare this run against the debug config to separate batch-size effects from dataset-order effects
