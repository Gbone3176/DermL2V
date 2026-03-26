# Residual MLP Pooling Hypothesis (V4)

## Context

- Motivation: explain why NV-Embed style latent pooling produced negative gain in the current dermatology fine-tuning workflow.
- New code:
  - `llm2vec/llm2vecV4.py`
  - `llm2vec/pooling_residual_mlp.py`

## Working Hypothesis

The current base encoder already has a reasonably good natural-text embedding space.
In this setting, NV-style latent pooling may hurt because:

- it adds a relatively large number of extra parameters
- it introduces randomly initialized latent prototypes
- token-to-latent cross attention injects noise at the beginning of training
- early optimization may severely distort the original embedding geometry

As a result, the model may lose more from representation destruction than it gains from extra adaptation capacity.

## V4 Design

Replace latent pooling with a lightweight residual MLP pooler:

- token representations stay in the original hidden space
- a 4-layer MLP predicts only a residual correction
- the residual branch is scaled by a small `gamma`
- the final linear layer is zero-initialized

This makes initialization close to plain mean pooling, so domain adaptation starts from "small correction" instead of "large rewrite".

## Why This Is A Good Test

If V4 performs better than latent pooling, then the likely issue is not "pooling needs more capacity", but rather:

- the adaptation branch was too disruptive
- random latent prototype interaction was too noisy
- preserving the pretrained embedding geometry matters more than adding expressive pooling structure

If V4 still fails in the same way, then the bottleneck is more likely elsewhere:

- data mixture conflict
- hard negative quality
- loss instability
- optimization schedule

## Recommended First Comparison

Run a short controlled comparison under the same:

- base model
- dataset mix
- seed
- batch size
- learning rate
- loss

Compare:

1. `mean`
2. `latent_pooling`
3. `res_mlp_pooling`

## Key Metrics To Watch

- validation retrieval metrics
- first 100-500 steps training stability
- `grad_norm`
- whether validation dip appears earlier or more severely
- cosine similarity distribution drift relative to the baseline encoder

## Additional Ablations Worth Trying

- V4 with `gamma_init=1e-3`
- V4 with `gamma_init=1e-2`
- V4 with and without output L2 normalization
- V4 with and without pooled output LayerNorm

## Decision Rule

- If V4 > latent pooling and close to or better than mean pooling:
  the "latent structure is too destructive" hypothesis becomes much more credible.
- If V4 < mean pooling but > latent pooling:
  lightweight correction helps, but the main bottleneck may still be data/loss.
- If V4 and latent pooling both underperform mean pooling:
  extra pooling capacity is probably not the right lever right now.
