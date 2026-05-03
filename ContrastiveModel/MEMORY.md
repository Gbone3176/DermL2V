# ContrastiveModel Memory

## Scope

- `ContrastiveModel/` is the isolated workspace for contrastive-model fine-tuning.
- All scripts, logs, downstream evaluation outputs, summaries, and other artifacts related to contrastive-model fine-tuning or comparison experiments must stay under `ContrastiveModel/`.
- Do not write contrastive-model fine-tuning or downstream-evaluation products into the repository-level zero-shot output folders such as `output/downstream/RT_text/nonhomo-full/`.
- Do not modify existing repository scripts for these experiments unless the user explicitly changes this rule.
- Each target model must have its own directory, output root, logs, and model-specific entry scripts.
- Shared data and loss interfaces are allowed under `ContrastiveModel/shared/` when they avoid duplication and do not couple model outputs together.

## Model Plan

- First-stage fine-tuning targets:
  - `NeuML/pubmedbert-base-embeddings`
  - `emilyalsentzer/Bio_ClinicalBERT`
- BERT-based models can use full-parameter fine-tuning.
- Later large-model targets should initially use LoRA or QLoRA:
  - `BMRetriever/BMRetriever-7B`
  - `Qwen/Qwen3-Embedding-8B`
  - `nvidia/NV-Embed-v2`

## Feasibility Notes

- `NeuML/pubmedbert-base-embeddings` is already an embedding-oriented PubMedBERT checkpoint and is the simplest model to domain-adapt on DermVariants.
- `emilyalsentzer/Bio_ClinicalBERT` is a BERT backbone, not a ready-made retrieval embedding model. For this project, fine-tune it with the CLS token hidden state plus L2 normalization as the embedding output.
- `BMRetriever-7B` is retrieval-oriented and can be adapted with LoRA/QLoRA later, preserving query/document instruction conventions.
- `BMRetriever-7B` LoRA fine-tuning lives under `ContrastiveModel/BMRetriever7B/`; use fp16 on the local V100 machine, last-token pooling with explicit EOS, BMRetriever-style query/passage instructions, and dot-product InfoNCE with row-aligned hard negatives.
- `Qwen3-Embedding-8B` should use the `qwen3` conda environment because it needs newer `transformers` support.
- `NV-Embed-v2` is trainable in principle but has higher engineering risk because it uses custom model code and a nonstandard pooling stack.

## Local Defaults

- BERT-based model experiments should use the `l2v` conda environment by default.
- Qwen3 or modern embedding models should use the `qwen3` conda environment by default.
- DermVariants training data:
  - `/storage/dataset/dermatoscop/Derm1M/DermVariantsData`
- The DermVariants JSONL files contain:
  - `original`
  - `positive_variant`
  - `hard_negative_variant`
- PubMedBERT follows the existing DermL2V-style query instruction convention.
- BioClinicalBERT fine-tuning uses raw query, positive, and hard-negative text with no instruction and no separator prefix on either side.
- The existing separator convention is `!@#$%^&*()`.

## Training Objective

- Use row-aligned triplets: query, positive, hard negative.
- Use in-batch positives as the main candidate pool.
- Keep explicit hard negatives row-aligned to the matching query rather than sharing every hard negative across all queries.
- Prefer the original model's training-stability settings when known, especially pooling mode, similarity scale, optimizer shape, and max sequence length.
- Use cosine similarity with the model's original temperature/scale value when known. For `NeuML/pubmedbert-base-embeddings`, default to `20.0`.
- Normalize embeddings before similarity scoring for stable retrieval training and downstream comparison.
- For DDP training, compute effective batch as `per_device_batch_size * gradient_accumulation_steps * num_gpus`.
- All contrastive fine-tuning runs must log training loss to SwanLab project requested as `Contrastive Model fine-tune`.
- SwanLab project identifiers do not allow spaces, so scripts sanitize that requested name to `Contrastive-Model-fine-tune` for the actual API call.
- All contrastive fine-tuning runs must evaluate retrieval on the validation split and log `NDCG@10` and `Recall@10` to the same SwanLab run.
- PubMedBERT formal training uses two GPUs (`4,6`), per-device batch `64`, global micro-batch `128`, gradient accumulation `4`, effective batch `512`, and `2` epochs.
- PubMedBERT formal training saves intermediate checkpoints every `25` optimizer steps, plus a final checkpoint at completion.
- BioClinicalBERT formal training uses GPUs `2,5,6,7`, per-device batch `64`, gradient accumulation `2`, effective batch `512`, CLS pooling, and raw text formatting by default.
- Remove obsolete tmux sessions promptly after experiments are stopped, superseded, or verified finished. Keep only tmux sessions that still host an active useful process or are explicitly needed for immediate inspection.

## Directory Contract

- `shared/`: reusable local interfaces for dataset loading, pooling, and losses.
- `PubMedBERT/`: PubMedBERT-only training, run, and loading notes.
- `BioClinicalBERT/`: BioClinicalBERT-only training, run, and loading notes.
- `BMRetriever7B/`: BMRetriever-7B-only LoRA training, run, and loading notes.
- Large-model directories should be added later without changing the BERT directories.
