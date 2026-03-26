# Experiment Retrospectives

This directory is for recording experiment outcomes, failures, successes, and follow-up decisions.

Subfolders:

- `data/`: data quality, sampling, hard negative, split, and label analysis
- `methods/`: architecture, optimization, pooling, loss, and evaluation behavior

Suggested structure:

- One markdown file per experiment or issue
- File name pattern:
  - `YYYY-MM-DD_short_title.md`
  - Example: `2026-03-18_v1_v4_validation_dip.md`

Recommended template:

```md
# Title

## Context
- Experiment name:
- Config:
- Code path:
- Date:

## What Happened
- Main result:
- Key curves / metrics:
- Unexpected behavior:

## Likely Causes
- Cause 1
- Cause 2
- Cause 3

## Evidence
- Plot:
- Log:
- Script:
- Dataset notes:

## Decision
- Keep / stop / revise:
- Best checkpoint:
- Next action:

## Follow-up
- [ ] Action 1
- [ ] Action 2
- [ ] Action 3
```

Current focus ideas:

- `data/`: `DermVariantsData` quality, hard negative behavior, margin distributions
- `methods/`: epoch-boundary effects, V1 vs V4 validation behavior, optimization dynamics
