# TopK-Shared SlerpMixCSE (HardNegativeNLLLossV5)

## Context
- Config: `train_configs/topk_shared_slerp_mixcse/Llama31-8b-inst-mntp-supervised_TopKSharedSlerpMixCSE@DermVariantsSFT.json`
- Runner: `experiments/run_supervised_with_eval.py`
- Command: `train_configs/topk_shared_slerp_mixcse/commands_topk_shared_slerpmixcse.sh`

## Method Intent
- Build mixed negatives as in V4, then share across batch and select per-query top-k (`shared_mix_topk`).
- Intended to enrich hard distractors without using all mixed negatives.

## Plausible Failure Reasons
- Shared top-k can introduce cross-sample contamination: hard negatives relevant to one query may be semantically mismatched noise for another.
- Top-k from noisy mixed pool may increase false-hard pressure.
- Additional retrieval over mixed pool adds complexity without guaranteeing cleaner supervision.

## Decision
- Use only with strict subset-level diagnostics and ablation against V4.

## Next Checks
- [ ] Sweep small `shared_mix_topk` values with short-run budget.
- [ ] Measure whether top-k shared negatives increase near-miss false negatives.
