# Methods Inventory (LLM2Vec Retrospective)

## Scope
This index tracks the method-level attempts already implemented in this repo, so we can stop blind retries and move to evidence-driven iteration.

## Method Files
- `2026-03-18_llm2vec_core_modifications.md`
- `2026-03-18_latent_pooling_designs.md`
- `2026-03-18_loss_versions_v0_to_v5.md`
- `2026-03-18_focalmixcse_v3.md`
- `2026-03-18_slerp_mixcse_v4.md`
- `2026-03-18_topk_shared_slerp_mixcse_v5.md`
- `2026-03-18_layer_fusion_pooling.md`
- `2026-03-18_training_free_token_pretend.md`

## Priority For Next Round
1. Lock one reproducible baseline (`HardNegativeNLLLoss` or V1-style MixCSE) and re-run with subset-level metrics.
2. Validate data-side margin quality before changing model/loss again.
3. Run cheap ablations first (single epoch or capped steps) before any 30h full run.
