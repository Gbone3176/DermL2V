# NV-Embed-v2 DermVariants Finetuning

This directory contains the NV-Embed-v2 contrastive fine-tuning pipeline for DermVariants.

Current entrypoints:

- `train_nvembed2_lora.py`: current repository-style training script.
- `run_train.sh`: launch script using the `l2v` conda environment.

Legacy package files under `src/nvembed2_derm_ft/` are kept for reference, but the current pipeline should use the top-level training script.

The current pipeline uses:

- LoRA on `model.embedding_model`.
- Full training of `model.latent_attention_model`.
- NV-Embed query instructions: `Instruct: {task_definition}\nQuery: {query}`.
- Empty passage instruction, matching NV-Embed retrieval usage.
- Pool masks that exclude query instruction tokens from latent-attention pooling.
- Row-aligned hard-negative InfoNCE with in-batch positives.
- SwanLab logging for train loss and validation `NDCG@10` / `Recall@10`.
- Output under `ContrastiveModel/nvembed2/output`.

## Layout

- `train_nvembed2_lora.py`: current DDP-capable LoRA + latent-attention training entrypoint.
- `run_train.sh`: current launch script.
- `train.py`, `configs/`, `scripts/`, `src/nvembed2_derm_ft/`: legacy standalone package.

## Assumptions

- Base model weights already exist locally:
  `/cache/modelscope/models/nv-community/NV-Embed-v2`
- DermVariants data follows the same jsonl layout as `llm2vec`:
  `SemVariants_train.jsonl`, `VisVariants_train.jsonl`, `DermQA_train.jsonl`, `SI1_train.jsonl`, `SI2_train.jsonl`
- Each row contains:
  `original`, `positive_variant`, `hard_negative_variant`

## Train

```bash
cd /storage/BioMedNLP/llm2vec
bash ContrastiveModel/nvembed2/run_train.sh
```

## Important choices

- Default LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Default precision: `fp16`
- Default attention backend: model-native eager path
- Default pooling behavior: latent attention with query instruction tokens masked out.
- Default trainable scope: LoRA adapters plus full latent-attention module.
- Default learning rate: `1e-5` for all trainable parameters.
- Use the `l2v` environment. NV-Embed-v2 remote code is not compatible with the newer `transformers` in the `qwen3` environment.
- Single-V100 pressure testing at `max_length=512` found `per_device_batch_size=7` succeeds and `8` OOMs.
- The default 4-GPU launch uses `per_device_batch_size=4` and `gradient_accumulation_steps=32`, giving exact global batch size `4 * 4 * 32 = 512`.

## Output

Each checkpoint saves:

- LoRA adapter for `model.embedding_model` under `adapter/`
- latent attention state under `latent_attention/`
- tokenizer files
- `embedding_config.json`
- `trainer_state.json`

The adapter is attached to `model.embedding_model`; `latent_attention/` must also be restored for the fine-tuned checkpoint.

## Load finetuned adapter later

Typical flow:

1. Load base NV-Embed-v2 from the local snapshot with `trust_remote_code=True`
2. Attach the saved LoRA adapter to `model.embedding_model`
3. Load the saved `latent_attention/` weights into `model.latent_attention_model`
4. Run normal embedding inference through the full NV-Embed model

## Notes for V100

- Do not assume `bfloat16`
- `flash_attention_2` is not enabled here by default
- Start with `max_length=512`, `per_device_batch_size=1`, `gradient_accumulation_steps=16`
- Enable gradient checkpointing unless throughput is more important than memory
