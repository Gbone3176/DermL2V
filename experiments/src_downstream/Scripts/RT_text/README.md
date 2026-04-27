# RT Text

This is the single home for RT downstream code. The old sibling directory
`experiments/src_downstream/rt_text/` has been removed to avoid split ownership.

## Layout

- `configs/`: RT evaluation configs.
- `docs/`: task logic notes and run documentation.
- `src/`: importable RT evaluation implementations.
- `launch/`: active Python entrypoints used by manual wrappers and automated checkpoint sweeps.
- `manual/`: small shell wrappers for direct manual testing.
- `lib/`: shared launch-time helpers.
- `summary/`: markdown/table renderers.
- `plots/`: plotting utilities.
- `archive/`: reserved for retired RT files; keep empty unless something is intentionally archived.

## Active RT-full Entry

```bash
/opt/conda/envs/l2v/bin/python experiments/src_downstream/Scripts/RT_text/launch/sweep_checkpoints.py \
  --config-path experiments/src_downstream/Scripts/RT_text/configs/derml2v_loss02_rt_full_eval_paths.json \
  --method-key derml2vloss02_sm_lr1e4_k16_lerp \
  --mode full \
  --devices 0
```

RT-full model modules are called through:

```bash
/opt/conda/envs/l2v/bin/python -m experiments.src_downstream.Scripts.RT_text.src.nonhomo_full.nonhomo_RT_l2v_multi_full
```
