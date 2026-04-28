# Loss Functions Overview

This document describes the purpose of each `HardNegativeNLLLoss` variant in this
folder. Historical original losses and V0_2-based losses are separated because
they use different explicit-negative semantics.

## Original / Historical Losses

These files are preserved as historical implementations. They should be treated
as archived behavior unless an experiment explicitly needs to reproduce an old
run.

| Loss | Function / Role |
| --- | --- |
| `HardNegativeNLLLoss` | Current historical default alias. Implements MixCSE-style hard-negative NLL: in-batch positives, global explicit negatives, one row-wise mixed negative. |
| `HardNegativeNLLLossV0` | Baseline supervised contrastive NLL. Candidate pool is `[all positives | all explicit negatives]`; every explicit negative is shared across all queries. |
| `HardNegativeNLLLossV0_1` | V0 plus duplicate false-negative masking. Explicit negatives that exactly match positives are removed from the softmax candidates. |
| `HardNegativeNLLLossV0_2` | V0 with row-aligned explicit negatives. `neg_i` only contributes to `query_i`; in-batch positives remain shared. This is the new base semantics for future loss discussion. |
| `HardNegativeNLLLossV0_3` | V0 plus multi-positive handling for duplicate negatives. Duplicate false negatives are promoted into the positive set instead of being masked. |
| `HardNegativeNLLLossV0StructuredSelfAttnAblation` | V0 retrieval loss plus optional structured self-attention auxiliary loss. Used for historical SA-only ablations without SlerpMixCSE. |
| `HardNegativeNLLLossV1` | Fixed-lambda MixCSE. Selects each query's hardest negative from the global explicit-negative pool, builds one mixed negative, and appends it as a row-wise candidate. |
| `HardNegativeNLLLossV2` | V1 plus focal-style sample reweighting. Harder samples receive larger loss weights through `gamma`. |
| `HardNegativeNLLLossV3` | V1 plus dynamic `lam_i` and a mixed-negative margin penalty. The mix ratio depends on hard-negative difficulty. |
| `HardNegativeNLLLossV4` | SlerpMixCSE. Extends V1 by supporting `lerp` or `slerp` interpolation between positive and mined hard negative. |
| `HardNegativeNLLLossV5` | Top-k shared SlerpMixCSE. Builds one mixed negative per row, shares the mixed pool across the batch, and keeps top-k mixed candidates per query. |
| `HardNegativeNLLLossV6` | V5 plus optional auxiliary loss. This was the historical main loss for SA + SM / SlerpMixCSE experiments. |
| `HardNegativeNLLLossV7AnglE` | V6 plus AnglE-style or cosine-angle hybrid similarity. Used for angle-similarity SA/SM experiments. |

## V0_2-Based Losses

These are the preferred losses for new work. The raw explicit negative semantics
are always V0_2-based:

- `d_reps_neg` must be empty or row-aligned with `q_reps`.
- `neg_i` only participates in `query_i`'s raw negative column.
- Raw explicit negatives are not shared across queries.
- Mixed negatives are generated from row-aligned `(pos_i, neg_i)`.
- If `neg_i == pos_i`, the raw negative and its derived mixed candidate are masked.

| Loss | Function / Role |
| --- | --- |
| `HardNegativeNLLLossV0_2StructuredSelfAttnAblation` | V0_2 retrieval loss plus optional structured self-attention auxiliary loss. This is the V0_2 replacement for SA-only ablations. |
| `HardNegativeNLLLossV1_2` | Fixed-lambda MixCSE on V0_2. Builds `mixed_i = lam * pos_i + (1-lam) * neg_i` and appends it only for `query_i`. |
| `HardNegativeNLLLossV2_2` | V1_2 plus focal-style sample reweighting through `gamma`. |
| `HardNegativeNLLLossV3_2` | Dynamic-lambda MixCSE on V0_2. Computes `lam_i` from `sim(q_i, pos_i)` and `sim(q_i, neg_i)`, then adds a mixed-negative margin penalty. |
| `HardNegativeNLLLossV4_2` | V0_2 SlerpMixCSE. Uses row-aligned `(pos_i, neg_i)` and supports `lerp` or `slerp` interpolation. |
| `HardNegativeNLLLossV5_2` | V0_2 top-k shared mixed-pool loss. Raw negatives stay row-private; generated mixed candidates form a shared pool, and each query keeps top-k mixed candidates. |
| `HardNegativeNLLLossV5_2_2` | V5_2 variant that always includes each query's own row-aligned mixed negative, then fills the remaining mixed slots with top-k non-own shared mixed candidates. |
| `HardNegativeNLLLossV6_2` | V5_2 plus optional auxiliary loss. This is the V0_2 replacement for SA + SM / SlerpMixCSE experiments. |
| `HardNegativeNLLLossV6_2_2` | V6_2 variant with the V5_2_2 mixed-candidate policy: own mixed negative is always included, with remaining mixed slots filled by top-k non-own shared mixed candidates. |
| `HardNegativeNLLLossV7_2AnglE` | V6_2 plus AnglE-style or cosine-angle hybrid similarity. This is the V0_2 replacement for angle-similarity SA/SM experiments. |

## Helper Module

| File | Function / Role |
| --- | --- |
| `HardNegativeNLLLossV0_2Common.py` | Shared helper module for V0_2-based losses. It is not registered as a trainable loss. It provides DDP gather, row-aligned negative validation, row-wise logits, duplicate masking, and `lerp` / `slerp` mix helpers. |

## Selection Guide

Use these defaults for new experiments:

| Experiment Type | Preferred Loss |
| --- | --- |
| Baseline row-aligned retrieval | `HardNegativeNLLLossV0_2` |
| SA-only / no SlerpMixCSE ablation | `HardNegativeNLLLossV0_2StructuredSelfAttnAblation` |
| Basic fixed-lambda MixCSE | `HardNegativeNLLLossV1_2` |
| SlerpMixCSE, one mixed candidate per row | `HardNegativeNLLLossV4_2` |
| Top-k shared SlerpMixCSE | `HardNegativeNLLLossV5_2` |
| Top-k shared SlerpMixCSE with forced own mixed negative | `HardNegativeNLLLossV5_2_2` |
| SA + top-k shared SlerpMixCSE | `HardNegativeNLLLossV6_2` |
| SA + top-k shared SlerpMixCSE with forced own mixed negative | `HardNegativeNLLLossV6_2_2` |
| SA + top-k shared SlerpMixCSE + AnglE similarity | `HardNegativeNLLLossV7_2AnglE` |
