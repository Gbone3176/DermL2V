# Training-Free Token Pretend / Separator-Mask Method

## Context
- Main code path:
  - `llm2vec/llm2vecV1.py` (`tokenize_with_separator`, `embed_mask`, `_skip_instruction`)
  - `experiments/run_supervised_with_eval.py` (`_encode_texts` requires separator)
- Typical separator: `!@#$%^&*()`

## Method Intent
- Without changing model weights, isolate content tokens from instruction tokens via separator and `embed_mask`.
- Pooling then focuses on content region, reducing instruction leakage.

## Plausible Failure Reasons
- Strong format dependency: missing separator causes hard failure or degenerate masks.
- Two-tokenization path (`original_texts` vs content-only ids) can be brittle under truncation and chat-template effects.
- If query/document prompt structure is inconsistent across subsets, masking logic may not align with actual semantic target.
- Training-free masking cannot resolve label noise or hard-negative ambiguity in dataset.

## Key Failure Signals To Track
- Zero/near-zero effective content length after masking.
- Masked length distribution drift across subsets.
- Performance sensitivity to separator placement and text pre-processing.

## Decision
- Keep as useful engineering trick, but not as core solution for current underperformance.

## Next Checks
- [ ] Add runtime stats: mask coverage ratio per batch/subset.
- [ ] Validate separator presence/position at dataset loader level.
- [ ] Run small ablation: `skip_instruction=True` vs `False` under identical checkpoints.
