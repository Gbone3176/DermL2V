# PubMedBERT RT Nonhomo-Full Evaluation

This pipeline evaluates all fine-tuned PubMedBERT checkpoints on the RT nonhomo-full datasets.

Defaults:

- Run directory: `ContrastiveModel/PubMedBERT/output/20260429_143552_pubmedbert-base-embeddings_pool-mean_b64_ga4_ep2.0_lr2e-05_scale20`
- Datasets: DermaSynth-E3, MedMCQA, MedQuAD, SCE-Derma-SQ
- Pooling: loaded from `embedding_config.json`, currently mean pooling
- Query input: `instruction + !@#$%^&*() + query`
- Document input: `!@#$%^&*() + document`
- Output root: `ContrastiveModel/PubMedBERT/rt_full_eval/output`

Run:

```bash
bash ContrastiveModel/PubMedBERT/rt_full_eval/run_pubmedbert_rt_full.sh
```

The script skips completed checkpoint-dataset JSON files unless `--overwrite` is passed to `eval_pubmedbert_rt_full.py`.
