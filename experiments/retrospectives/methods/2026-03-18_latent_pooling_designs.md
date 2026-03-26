# Latent Pooling Designs (V0/V1/V2/V3)

## Context
- Files:
  - `llm2vec/pooling_latent_V0.py`
  - `llm2vec/pooling_latent_V1.py`
  - `llm2vec/pooling_latent_V2.py`
  - `llm2vec/pooling_latent_V3.py`
  - `llm2vec/pooling_latent.py`

## Design Evolution
- V0: basic latent dictionary + MHA + MLP + mean pool.
- V1: adds LayerNorm and residual path.
- V2: NV-Embed style PreNorm + cross-attn + FFN + residual.
- V3: V2 plus SDPA/flash backend compatibility and dropout refinements.
- `pooling_latent.py`: practical branch with safer init and DDP-safe device casting.

## Potential Failure Reasons
- Objective mismatch: latent pooling adds capacity, but if bottleneck is noisy hard negatives/data conflict, capacity increase may not help.
- Optimization burden: additional pooling params can destabilize early training under high LR/large effective batch.
- Interface confusion: experiments may think they use V2/V3 while runtime uses `pooling_latent.py` through current import chain.

## Risk Notes
- Earlier V0/V1 patterns included in-forward device moves (`self.to(device)`), which are fragile in distributed training.
- Latent module only helps if attention mask semantics (`embed_mask` vs `attention_mask`) are correct and consistent.

## Decision
- Treat latent pooling as secondary lever; prioritize data/loss diagnostics first.

## Next Checks
- [ ] Add explicit runtime print of actual latent pooling module file/version.
- [ ] Compare mean vs latent_pooling under identical seed/data order and short budget.
