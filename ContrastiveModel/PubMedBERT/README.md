# PubMedBERT Contrastive Fine-Tuning

This directory is isolated for `NeuML/pubmedbert-base-embeddings` fine-tuning.

## Defaults

- Environment: `l2v`
- Model: `NeuML/pubmedbert-base-embeddings`
- Data: `/storage/dataset/dermatoscop/Derm1M/DermVariantsData`
- Objective: row-aligned hard-negative in-batch NLL
- Pooling: mean pooling
- Similarity scale: 20.0, matching the original model training config
- Fine-tuning: full-parameter BERT fine-tuning
- SwanLab requested project: `Contrastive Model fine-tune`
- SwanLab actual project id: `Contrastive-Model-fine-tune`
- Eval: validation-split retrieval, logging `NDCG@10` and `Recall@10`
- Formal training: GPUs `4,6`, per-device batch `64`, global micro-batch `128`, gradient accumulation `4`, effective batch `512`, `2` epochs
- Checkpointing: save intermediate weights every `25` optimizer steps, plus final weights at completion

## Single-GPU Smoke Run

```bash
conda run -n l2v python ContrastiveModel/PubMedBERT/train_pubmedbert.py \
  --max_train_samples 64 \
  --num_train_epochs 1 \
  --per_device_batch_size 8 \
  --output_root ContrastiveModel/PubMedBERT/debug_runs
```

## Multi-GPU Training

```bash
conda run -n l2v torchrun --nproc_per_node 4 ContrastiveModel/PubMedBERT/train_pubmedbert.py \
  --num_train_epochs 2 \
  --per_device_batch_size 64 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --loss_scale 20.0 \
  --save_steps 25 \
  --local_files_only \
  --gradient_checkpointing \
  --output_root ContrastiveModel/PubMedBERT/output
```

With two GPUs, `per_device_batch_size=64` gives a global micro-batch of `128`; `128 * 4 = 512`.

The script writes the resolved training arguments, encoder weights, tokenizer files, and embedding pooling config into the run directory.
