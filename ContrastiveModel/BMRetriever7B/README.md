# BMRetriever-7B LoRA Fine-Tuning

This directory is isolated for `BMRetriever/BMRetriever-7B` LoRA fine-tuning on DermVariants.

## Defaults

- Environment: `l2v`
- Local model: `/cache/transformers_cache/models--BMRetriever--BMRetriever-7B/snapshots/13e6adb9273c5f254e037987d6b44e9e4b005b9a`
- Data: `/storage/dataset/dermatoscop/Derm1M/DermVariantsData`
- Objective: dot-product InfoNCE with in-batch positives and one row-aligned hard negative
- Query format: task instruction, newline, `Query: {text}`
- Passage format: `Represent this passage`, newline, `passage: {text}`
- Pooling: last token after explicitly appending EOS
- Fine-tuning: LoRA, fp16
- LoRA defaults: `r=16`, `alpha=32`, dropout `0.05`, targets `q_proj,k_proj,v_proj,o_proj`
- Training start point: lr `1e-5`, max length `512`, temperature `1.0`
- SwanLab requested project: `Contrastive Model fine-tune`
- SwanLab actual project id: `Contrastive-Model-fine-tune`
- Eval: validation-split retrieval, logging `NDCG@10` and `Recall@10`

## Single-GPU Smoke Run

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n l2v python ContrastiveModel/BMRetriever7B/train_bmretriever7b_lora.py \
  --local_files_only \
  --disable_swanlab \
  --max_train_samples 4 \
  --eval_max_samples 4 \
  --max_steps 1 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_length 128 \
  --eval_batch_size 1 \
  --output_root ContrastiveModel/BMRetriever7B/debug_runs
```

## Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 conda run -n l2v torchrun --nproc_per_node 4 \
  ContrastiveModel/BMRetriever7B/train_bmretriever7b_lora.py \
  --local_files_only \
  --max_length 512 \
  --num_train_epochs 1 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --temperature 1.0 \
  --save_steps 25 \
  --eval_batch_size 4 \
  --gradient_checkpointing \
  --output_root ContrastiveModel/BMRetriever7B/output
```

With four GPUs, `per_device_batch_size=1` and `gradient_accumulation_steps=16` gives an effective batch size of `64`.

The script writes the resolved training arguments, run metadata, LoRA adapter, tokenizer files, and embedding config into the run directory.
