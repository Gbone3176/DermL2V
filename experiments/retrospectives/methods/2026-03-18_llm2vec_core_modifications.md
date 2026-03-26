# LLM2Vec Core Modifications

## Context
- Goal: extend base LLM2Vec to support instruction-aware masking, latent pooling, and additional pooling modes.
- Main code:
  - `llm2vec/llm2vecV1.py`
  - `llm2vec/llm2vecV3.py`
  - `experiments/run_supervised_with_eval.py`
  - `experiments/run_supervised_fusion_withEval.py`

## What Was Changed
- V1 branch used in `run_supervised_with_eval.py` imports `llm2vec.llm2vecV1.LLM2Vec`.
- V1 supports `mean/weighted_mean/eos/last/bos/latent_pooling`, plus `skip_instruction` and `embed_mask` flow.
- V3 branch adds `pooling_mode="layer_fusion"` and fusion-specific trainable modules.

## Why It Might Fail
- Implementation branch mismatch risk: different scripts use different LLM2Vec class versions.
- Method comparison can be confounded if one experiment is on V1 and another on V3.
- Pooling behavior and instruction masking are tightly coupled to separator formatting; data-format drift causes silent behavior change.

## Evidence Anchors
- `run_supervised_with_eval.py` imports `llm2vecV1`.
- `run_supervised_fusion_withEval.py` imports `llm2vecV3`.
- `llm2vec/__init__.py` default export points to `llm2vec.py`, not V1.

## Decision
- Keep both branches, but always record script + class version in experiment metadata.

## Next Checks
- [ ] Add a run header log: LLM2Vec class path + git commit + loss class.
- [ ] Add a guard in configs to prevent accidental V1/V3 cross-comparison without explicit flag.
