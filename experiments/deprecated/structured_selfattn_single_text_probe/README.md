# Structured Self-Attn Single-Text Probe

This probe loads the DermL2V SA+SM checkpoint, samples one test example from the CLS-skincap test set, and checks:

1. Whether the `r` structured self-attention hops produce different hop-level representations.
2. How the final structured self-attention embedding differs numerically from mean pooling on the same hidden states and mask.

Default assumptions:

- checkpoint: `checkpoint-50` under `SlerpMixCSE_k128_StructuredSelfAttn_gamma0p001_aux0p001`
- test set: `/storage/dataset/dermatoscop/DermEmbeddingBenchmark/exp2-skincap-DiseaseClassification/test.jsonl`
- input format: `instruction + "!@#$%^&*()" + sentence`
- `skip_instruction=True`

Run:

```bash
python experiments/deprecated/structured_selfattn_single_text_probe/run_probe.py
```

Outputs:

- `probe_result.json`
- `probe_summary.md`
