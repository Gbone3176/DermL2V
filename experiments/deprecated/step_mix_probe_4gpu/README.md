# Step Mix Probe

This experiment reconstructs the real `DermVariants` training traversal under the intended optimizer-step settings:

- `per_device_train_batch_size = 64`
- `per_device_eval_batch_size = 64`
- `gradient_accumulation_steps = 8`
- `num_processes = 4`
- `do_train = true`

It exports the real sample stream for steps `50-80`, plus per-step and aggregate dataset-source ratios.

## Run

```bash
/opt/conda/envs/l2v/bin/python experiments/deprecated/step_mix_probe_4gpu/run_probe.py
```

## Outputs

- `outputs/step_50_80_summary.csv`
- `outputs/step_50_80_rank_microbatch_summary.csv`
- `outputs/step_50_80_samples.jsonl`
- `outputs/step_50_80_aggregate.json`
- `outputs/step_50_80_report.md`
