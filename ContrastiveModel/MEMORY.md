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
- Resolve machine-specific dataset and model paths through `local_info/`, especially `local_info/local_path.md` and `local_info/models_info.md`.
- Do not hard-code machine-local absolute paths into reusable training scripts or workflow notes.
- DermVariants training data is referred to as `DermVariantsData`; the concrete local path belongs in `local_info/`.
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

## Fine-Tuning Workflow

Use this workflow when moving the contrastive-model fine-tuning code to another machine or launching a new comparison-model run.

1. Prepare machine-local references.
   - Check `local_info/models_info.md` for model identifiers or local model-cache entries.
   - Check `local_info/local_path.md` for dataset and adapter roots.
   - Keep concrete machine paths in `local_info/`; scripts should expose them as environment overrides or documented defaults.

2. Choose the correct model directory.
   - PubMedBERT: `ContrastiveModel/PubMedBERT/`
   - BioClinicalBERT: `ContrastiveModel/BioClinicalBERT/`
   - BMRetriever-7B: `ContrastiveModel/BMRetriever7B/`
   - Qwen3-Embedding-8B: `ContrastiveModel/Qwen3Embedding8B/`
   - NV-Embed-v2: `ContrastiveModel/nvembed2/`
   - Shared dataset/loss/model helpers: `ContrastiveModel/shared/`

3. Start with a smoke test.
   - Use a short run before full training.
   - Use `max_steps` or the directory's smoke script where available.
   - Disable final checkpoint saving during pure capacity or stability probes when supported.
   - Confirm that model loading, data loading, first optimizer steps, loss logging, and checkpoint or skip-save behavior match expectation.
   - Treat any `nan`, `inf`, CUDA OOM, NCCL timeout, or checkpoint shape mismatch as a blocker before full training.

4. Determine memory-safe batch shape.
   - Record `per_device_batch_size`, `gradient_accumulation_steps`, number of GPUs, `max_length`, precision mode, and effective batch size.
   - Effective batch size is `per_device_batch_size * gradient_accumulation_steps * num_gpus`.
   - The comparison target for formal runs is usually effective batch size `512`, unless the user explicitly chooses another target.
   - If a model cannot fit the desired per-device batch, reduce per-device batch and compensate with gradient accumulation.

5. Launch formal training in tmux.
   - Use a short readable tmux session name tied to the model and run.
   - Keep the session available after launch for inspection.
   - Do not delete tmux sessions just because training finished unless cleanup is explicitly requested or the session is stale and no longer useful.
   - Put training outputs, logs, debug runs, probes, and checkpoints under the model directory inside `ContrastiveModel/`.
   - Do not put comparison-model fine-tuning outputs under repository-level zero-shot output folders.

6. Monitor training.
   - Check GPU memory and utilization shortly after launch.
   - Confirm loss appears after the first optimizer step.
   - Watch the first few optimizer steps carefully; many numerical failures appear immediately after the first update.
   - Confirm intermediate checkpoints are written at the configured save interval.
   - If a run is manually stopped, record the latest complete checkpoint and whether a final checkpoint exists.

7. Run RT nonhomo-full evaluation.
   - Use the model-specific `rt_full_eval/` scripts when present.
   - Keep evaluation outputs under the corresponding model's `rt_full_eval/output/` directory.
   - Summaries should include checkpoint rows, zero-shot baseline rows when available, percent-formatted `NDCG@10` and `Recall@10`, average columns, and a best-checkpoint line.
   - Do not overwrite repository-level zero-shot outputs; read them only as baselines.

8. Sync for another machine.
   - Commit and push only training-related code, configs, launch scripts, lightweight documentation, and shared helpers.
   - Do not commit checkpoints, `output/`, `logs/`, `debug_runs/`, `bs_probe/`, `resume_probe/`, `__pycache__/`, or generated tokenizer/model artifacts.
   - If evaluation or retrospective files are unrelated to the transfer goal, leave them uncommitted unless explicitly requested.

## Model-Specific Training Notes

- PubMedBERT:
  - Uses full-parameter fine-tuning.
  - Uses mean pooling by default for `NeuML/pubmedbert-base-embeddings`.
  - Formal local run used effective batch size `512`, intermediate checkpoints every `25` optimizer steps, and a final checkpoint.

- BioClinicalBERT:
  - Uses full-parameter fine-tuning.
  - Uses CLS pooling and raw text formatting by default.
  - Formal local run used effective batch size `512`, intermediate checkpoints every `25` optimizer steps, and a final checkpoint.

- BMRetriever-7B:
  - Uses LoRA fine-tuning.
  - Uses BMRetriever-style query/passage instructions, explicit EOS handling, last-token pooling, and row-aligned hard negatives.
  - The best RT nonhomo-full checkpoint observed so far was an early checkpoint, so do not assume later checkpoints are better without evaluation.

- Qwen3-Embedding-8B:
  - Uses LoRA fine-tuning in the `qwen3` environment.
  - The local V100 run was stopped early after the loss plateaued.
  - RT nonhomo-full improvement over the zero-shot baseline was small, so further training should be revisited on a larger-memory machine with room for better batch, sequence length, or optimizer experiments.

- NV-Embed-v2:
  - Uses LoRA on the embedding model plus trainable latent attention.
  - Latent attention should not be frozen by default because it is the model's pooling mechanism.
  - Current V100 fp16 training is numerically unstable: full training and `lr=1e-6` smoke tests produced `nan` shortly after the first optimizer update.
  - Do not treat the failed NV checkpoints as valid training products.
  - Next stability probes should test fp32 trainable modules, larger Adam epsilon, finite-value checks before and after optimizer updates, and smaller per-device batch sizes before returning to formal training.

## Completion Checklist

- Training script compiles.
- Smoke test reaches at least two optimizer steps without non-finite values.
- Full run writes expected checkpoints or is intentionally stopped at a known checkpoint.
- Logs identify model, batch shape, learning rate, temperature/scale, precision, and effective batch size.
- RT nonhomo-full summary includes zero-shot baseline if available.
- Best checkpoint is selected from evaluation results, not from latest checkpoint by default.
- Transfer commits include code and configs only, not local artifacts.
