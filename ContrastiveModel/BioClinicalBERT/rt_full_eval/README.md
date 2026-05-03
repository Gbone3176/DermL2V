# BioClinicalBERT RT Nonhomo-Full Evaluation

This pipeline evaluates all fine-tuned BioClinicalBERT checkpoints on the RT nonhomo-full datasets.

Defaults:

- Run directory: `ContrastiveModel/BioClinicalBERT/output/20260430_102154_bioclinicalbert_pool-cls_b64_ga2_ep2.0_lr2e-05_scale20_rawtext`
- Datasets: DermaSynth-E3, MedMCQA, MedQuAD, SCE-Derma-SQ
- Pooling: CLS pooling
- Query/document formatting: raw text, no instruction, no separator
- Output root: `ContrastiveModel/BioClinicalBERT/rt_full_eval/output`

Run:

```bash
bash ContrastiveModel/BioClinicalBERT/rt_full_eval/run_bioclinicalbert_rt_full.sh
```

The evaluator skips completed checkpoint-dataset JSON files unless `--overwrite` is passed through to the Python entrypoint.
