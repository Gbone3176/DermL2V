# Downstream Pipeline Scripts

This directory stores launch/config/orchestration files for DermL2V downstream testing.

## Layout

- `CLS_text/`
  - CLS configs, launch scripts, and DermL2V CLS pipeline drivers.
- `RT_text/`
  - RT launch scripts for homo and nonhomo evaluation.
- `pipelines/`
  - Cross-task orchestration scripts that coordinate RT and CLS stages.
  - `archive/` contains stale orchestration entrypoints that reference missing historical drivers.
- `deprecated/`
  - Historical task scripts kept for reference only.

Generated Python caches are intentionally excluded from this tree.
