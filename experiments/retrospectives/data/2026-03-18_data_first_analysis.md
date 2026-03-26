# Data-First Analysis

## Context
- Experiment family: `DermVariants` supervised training and retrieval validation
- Related configs:
  - `train_configs/supervised/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT.json`
  - `train_configs/supervised/MetaLlama3.1_8B_inst-mntp_supervised@DermVariantsSFT_Debug.json`
- Related code:
  - `llm2vec/dataset/DermVariants.py`
  - `experiments/run_supervised_withEval.py`
  - `experiments/v1_v4_validation/run_v1_v4_validation.py`
- Analysis date: `2026-03-18`

## Goal
- Explain model underperformance from the data side before changing model design again.

## What We Confirmed

### 1. The training set is a mixed-task dataset, not a single task
- `DermVariants.py` mixes five subsets into one training dataset:
  - `SemVariants`
  - `VisVariants`
  - `DermQA`
  - `SI1`
  - `SI2`
- This means the model is not optimizing for one retrieval definition.
- Different subsets have different answer styles, difficulty patterns, and text formats.

### 2. Formatting is inconsistent across subsets
- `SemVariants`, `VisVariants`, and `SI1` prepend the same instruction to both query and doc.
- `DermQA` and `SI2` prepend instruction only on the query side.
- This introduces task-style shortcuts and weakens the idea of a shared embedding space.

### 3. Training is dominated by `SI1` and `SI2`
- After `DermQA` upsampling x10, the effective training mixture is still mostly:
  - `SI2`
  - `SI1`
  - `VisVariants`
- `DermQA` remains a minority task.
- Therefore downstream gains may mostly reflect optimization toward long-answer clinical retrieval rather than the target benchmark.

### 4. Hard negatives are not equally clean across subsets
- Same mining logic does not produce the same quality across tasks.
- Main reason: task definitions differ.
- Observed pattern:
  - `SemVariants`: relatively reasonable hard negatives
  - `VisVariants`: positive and negative often nearly equally close to query
  - `SI1`: positive is often a short label, while negative is a fuller explanation
  - `DermQA`: positive and negative often answer adjacent sub-questions for the same disease
  - `SI2`: cleaner than `SI1`, but still contains topic-near negatives

### 5. `SI1` is structurally biased
- In many `SI1` samples:
  - query = long clinical vignette
  - positive = short label or short answer
  - negative = longer explanatory text
- This makes similarity-based learning unstable because the negative can be semantically richer than the positive.

### 6. `VisVariants` is intrinsically harder
- This subset is not simple paraphrase matching.
- It behaves more like diagnosis-style text to visual-description text alignment.
- Query-positive lexical overlap is naturally low.
- Hard negatives can become near-boundary cases even when the mining script is unchanged.

### 7. Structural data corruption has been partly cleaned
- `VisVariants_train.jsonl` had empty `positive_variant` and `hard_negative_variant` entries.
- These 18 invalid rows were removed.
- Current structural audit shows:
  - no missing required keys
  - no empty core text fields
  - no duplicate ids

## Important Quantitative Signals

### Approximate effective training mixture
- `SemVariants`: about 11.8%
- `VisVariants`: about 19.5%
- `SI1`: about 29.9%
- `SI2`: about 31.4%
- `DermQA`: about 7.5%

### Hard-negative warning signals
- `VisVariants` showed especially weak positive-vs-negative separation.
- In sampled analysis, many `VisVariants` negatives were as close or closer to the query than positives.
- This suggests label-boundary noise rather than purely useful hard negatives.

## Training Dynamics Relevance

### 1. Each optimizer step sees mixed subsets
- Data is globally shuffled, then chunked, then batch-order shuffled.
- Training does not run one subset fully before another.
- Every step contains a mixture of several subsets.

### 2. There is an epoch-boundary effect around step 67
- Under the debug config:
  - 4 GPUs
  - `per_device_train_batch_size = 64`
  - `gradient_accumulation_steps = 8`
- Effective samples per optimizer step:
  - `64 * 4 * 8 = 2048`
- Effective train size after batching alignment:
  - about `136960`
- Steps per epoch:
  - about `67`
- This likely explains:
  - local metric dip around `50-90`
  - `grad_norm` reaching a low around `65` then rising again after epoch rollover

## Current Interpretation
- The main bottleneck is not clearly the architecture.
- The data mixture itself likely injects conflicting objectives.
- Several subsets contain hard negatives that are too close, noisy, or unfair relative to the positive.
- Some tasks are mismatched to a shared contrastive formulation.

## Decisions
- Do not treat this as a pure model-capacity problem yet.
- Prioritize data diagnosis before adding new architectural complexity.

## Next Actions
- [ ] Analyze margin distributions for all five subsets
- [ ] Inspect lowest-margin samples per subset
- [ ] Audit `SI1` for short-positive / long-negative mismatch
- [ ] Audit `VisVariants` for near-duplicate or boundary-noisy negatives
- [ ] Compare train/validation subset-specific distributions
- [ ] Consider task-specific negative mining instead of one shared strategy
