# BioClinicalBERT Contrastive Fine-Tuning

This directory is isolated for `emilyalsentzer/Bio_ClinicalBERT` fine-tuning.

## Defaults

- Environment: `l2v`
- Model: `emilyalsentzer/Bio_ClinicalBERT`
- Local snapshot: `/cache/transformers_cache/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b`
- Data: `/storage/dataset/dermatoscop/Derm1M/DermVariantsData`
- Objective: row-aligned hard-negative in-batch NLL
- Pooling: CLS token hidden state
- Similarity scale: `20.0`
- Text formatting: no instruction and no separator prefix on either query or document side
- Fine-tuning: full-parameter BERT fine-tuning
- SwanLab requested project: `Contrastive Model fine-tune`
- SwanLab actual project id: `Contrastive-Model-fine-tune`
- Eval: validation-split retrieval, logging `NDCG@10` and `Recall@10`
- Formal training: GPUs `2,5,6,7`, per-device batch `64`, global micro-batch `256`, gradient accumulation `2`, effective batch `512`, `2` epochs
- Checkpointing: save intermediate weights every `25` optimizer steps, plus final weights at completion

## Single-GPU Smoke Run

```bash
conda run -n l2v python ContrastiveModel/BioClinicalBERT/train_bioclinicalbert.py \
  --max_train_samples 64 \
  --num_train_epochs 1 \
  --per_device_batch_size 8 \
  --output_root ContrastiveModel/BioClinicalBERT/debug_runs \
  --disable_swanlab
```

## Multi-GPU Training

```bash
conda run -n l2v torchrun --nproc_per_node 4 ContrastiveModel/BioClinicalBERT/train_bioclinicalbert.py \
  --num_train_epochs 2 \
  --per_device_batch_size 64 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --loss_scale 20.0 \
  --save_steps 25 \
  --local_files_only \
  --gradient_checkpointing \
  --output_root ContrastiveModel/BioClinicalBERT/output
```

The script writes the resolved training arguments, encoder weights, tokenizer files, and embedding pooling config into the run directory.
