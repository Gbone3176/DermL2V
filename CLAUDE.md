# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DermL2V** is a dermatology-specialized fork of [LLM2Vec](https://arxiv.org/abs/2404.05961), converting decoder-only LLMs into text encoders for dermatological text understanding. It extends the original LLM2Vec framework with:

1. **Latent attention pooling** — a cross-attention-based pooling mechanism using learnable latent vectors (replacing simple mean pooling)
2. **Multi-layer feature fusion** — a learned router that dynamically weights multiple final hidden layers
3. **MixCSE contrastive loss** with hard-negative mining, mixed negative construction, and optional focal-style reweighting
4. **Dermatology-specific datasets** covering semantic similarity, visual description matching, QA retrieval, and clinical vignettes
5. **Multiple PEFT adapter chaining** — loading and merging multiple LoRA adapters sequentially
6. **Comprehensive downstream evaluation** — retrieval tasks (homogeneous/non-homogeneous), text classification (SkinCAP), and BLURB biomedical benchmarks

The original LLM2Vec three-step pipeline is preserved:
1. Enabling bidirectional attention (modifying causal attention to non-causal)
2. Masked Next Token Prediction (MNTP) training
3. Unsupervised contrastive learning (SimCSE) or supervised contrastive training

## Installation

```bash
pip install -e .
pip install flash-attn==2.8.3 --no-build-isolation
```

Core requirements: `transformers>=4.43.1,<=4.44.2`, `torch==2.5.1+cu121`. Training additionally requires `accelerate`, `swanlab`, `peft`. Evaluation requires `beir`, `mteb`, `sentence-transformers`. See `requirements.txt` for full pinned dependencies.

## Key Commands

### Training

**Supervised contrastive with evaluation (primary training script):**
```bash
# Single GPU
python experiments/run_supervised_with_eval.py train_configs/supervised/<Config>.json

# Multi-GPU with torchrun
torchrun --nproc_per_node=<N> experiments/run_supervised_with_eval.py train_configs/supervised/<Config>.json
```

**Layer fusion training (backbone frozen, trains only fusion router):**
```bash
# Single GPU
python experiments/run_supervised_fusion_withEval.py train_configs/Baseline_MixCSE_Fusion/<Config>.json

# Multi-GPU
torchrun --nproc_per_node=<N> experiments/run_supervised_fusion_withEval.py train_configs/Baseline_MixCSE_Fusion/<Config>.json
```

**Basic supervised contrastive (no eval callback):**
```bash
torchrun --nproc_per_node=<N> experiments/run_supervised.py train_configs/supervised/<Config>.json
```

### Evaluation

**Retrieval evaluation on DermVariants test set:**
```bash
python experiments/test_supervised.py test_configs/supervised/<Config>.json
```

**Downstream text classification (SkinCAP Disease Classification):**
```bash
python -m BLURB-src.seqcls.run_seqcls_llm2vec <Args>
```

**Downstream retrieval tasks (homogeneous/non-homogeneous):**
```bash
python experiments/src_downstream/homo_RT_l2v.py --model_path <PATH> --data_path <DATA>
python experiments/src_downstream/nonhomo_RT_l2v.py --model_path <PATH> --data_path <DATA>
```

**Batch encoding to .npy files:**
```bash
python -m experiments.encode --model_path <PATH> --input_file <JSONL> --output_dir <DIR> --num_shards <N> --shard_id <ID>
```

**BLURB biomedical benchmark:**
```bash
bash BLURB-src/run_llm2vec_on_BLURB.sh
```

**Visualization (UMAP embedding plots / kNN purity):**
```bash
python visualization/embed_l2v_umap.py --model_path <PATH> --data_path <CSV>
python visualization/knn_l2v.py --model_path <PATH> --data_path <CSV>
```

## Architecture

### Core Class: `llm2vec/llm2vec.py` — `LLM2Vec`

`LLM2Vec` is an `nn.Module` wrapper around HuggingFace models. The current active version (V3) includes layer fusion support. Version history:

| Version | File | Default Pooling | Latent Pooling | Layer Fusion | Extra PEFT Chaining |
|---------|------|----------------|----------------|--------------|---------------------|
| V0 | `llm2vecV0.py` | mean | No | No | No |
| V1 | `llm2vecV1.py` / `llm2vec_bac.py` | latent_pooling | Yes | No | Yes |
| V3 (current) | `llm2vec.py` / `llm2vecV3.py` | latent_pooling | Yes | Yes | Yes |

Key responsibilities:
- **Model loading** via `from_pretrained()`: loads a base model, optionally applies and merges PEFT/LoRA adapters (supports chaining multiple adapters via `extra_model_name_or_path` as string or list), auto-loads `latent_attn.pt` or `layer_fusion_router.pt` weights
- **Bidirectional model selection**: maps config class names (LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config, Qwen3Config) to their `*BiModel` counterparts via `_get_model_class()`
- **Tokenization**: `prepare_for_tokenization()` wraps text with model-specific instruction templates (Llama3 chat format, Mistral `[INST]` tags, Gemma turn markers, Qwen2 `im_start/im_end` tags)
- **Encoding**: `encode()` produces embeddings from text inputs in the form `[[instruction, text], ...]` or `[text, ...]`; `encode_text()`, `encode_with_separator()`, `encode_with_instruction()` are convenience APIs
- **Similarity**: `compute_similarities()` computes cosine similarity between query and candidates
- **BERT format**: `convert_to_bert_format()` returns `last_hidden_state` + `pooler_output` dict for compatibility

**Pooling modes (7 total):**
1. `mean` — masked mean pooling over non-padding, non-instruction tokens
2. `weighted_mean` — linearly weighted mean (later tokens get higher weight)
3. `eos_token` / `last_token` — takes the last token embedding
4. `bos_token` — takes embedding at the BOS token position
5. `latent_pooling` — uses `LatentAttentionPooling` module with learnable latent dictionary
6. `layer_fusion` — multi-layer feature fusion with learned router (see below)

**Layer Fusion mechanism (`llm2vec.py`):**
- Takes the last K hidden states (configurable via `layer_fusion_num_layers`)
- Mean-pools each layer's hidden states (masked)
- Applies a LayerNorm + learned MLP router (`Linear -> GELU -> Linear -> scalar`) to produce per-layer scores
- Softmax with temperature produces attention weights over layers
- Fused output = `base_mean_pool + gamma * weighted_sum_of_layers`, where `gamma` is a small learnable scalar (default init 1e-3)
- When `layer_fusion_train_only=True`, the backbone is frozen and only router/norm/gamma are trained
- `freeze_backbone_for_fusion_training()` / `set_backbone_trainable()` toggle backbone gradients

### Latent Attention Pooling: `llm2vec/pooling_latent.py`

`LatentAttentionPooling(nn.Module)` — a cross-attention-based pooling mechanism:
1. **Trainable latent dictionary**: `nn.Parameter` of shape `(num_latents, d_model)`, default 512 latents
2. **Multihead cross-attention**: Query = hidden states, Key/Value = latent dictionary (tokens attend to latents)
3. **Residual + LayerNorm**: `hidden_states + LayerNorm(attn_output)`
4. **MLP**: `Linear(D,D) -> GELU -> Linear(D,D)` on the residual output
5. **Masked mean pooling** over the sequence dimension

### CXR Model: `llm2vec/modeling_llm2vec4cxr.py`

`LLM2Vec4CXRModel(PreTrainedModel)` — a standalone HuggingFace `PreTrainedModel` for chest X-ray text encoding. Wraps `LlamaBiModel` + `LatentAttentionPooling`. Hardcoded to Llama architecture, designed for deployment on HuggingFace Hub (e.g., `lukeingawesome/llm2vec4cxr`). No PEFT support.

### `llm2vec/llm2vec_wrapper.py`

`LLM2VecWrapper(LLM2Vec)` — an older subclass that first introduced convenience APIs (`encode_text`, `encode_with_separator`, `compute_similarities`). These methods were later migrated into the main `LLM2Vec` class in V1+. The wrapper's `_load_latent_attention_weights()` loads from `pytorch_model.bin` (full checkpoint), while newer versions use dedicated `latent_attn.pt` files. Hardcoded to Llama3 tokenization only.

### Bidirectional Models: `llm2vec/models/`

Each supported architecture has a `bidirectional_*.py` file that:
1. Subclasses attention classes (eager, flash_attention_2, sdpa) and sets `is_causal = False`
2. Subclasses the decoder layer to use modified attention
3. Provides `*BiModel` (for encoding) and `*BiForMNTP` (for masked next token prediction)
4. Overrides `_update_causal_mask()` to produce a zero mask instead of triangular

| Architecture | File | BiModel | BiForMNTP | Registered |
|-------------|------|---------|-----------|------------|
| Llama | `bidirectional_llama.py` | LlamaBiModel | LlamaBiForMNTP | Yes |
| Mistral | `bidirectional_mistral.py` | MistralBiModel | MistralBiForMNTP | Yes |
| Gemma | `bidirectional_gemma.py` | GemmaBiModel | GemmaBiForMNTP | Yes |
| Qwen2 | `bidirectional_qwen2.py` | Qwen2BiModel | Qwen2BiForMNTP | Yes |
| Qwen3 | `bidirectional_qwen3.py` | Qwen3BiModel | Qwen3BiForMNTP | No (commented out in `__init__.py`) |

Adding a new architecture requires creating a new `bidirectional_*.py` and registering it in `llm2vec/models/__init__.py` and `LLM2Vec._get_model_class()`.

### Datasets: `llm2vec/dataset/`

All datasets extend `llm2vec.dataset.dataset.Dataset` (a `torch.utils.data.Dataset` subclass) and return `TrainSample` objects containing `texts` (List[str]) and `label`. Dataset loading is centralized in `llm2vec/dataset/utils.py:load_dataset()`.

| Dataset Name | Class | Purpose | Data Format |
|-------------|-------|---------|------------|
| `E5` | `E5Data` | Supervised contrastive from echo-embeddings (14 sub-datasets) | `[query, positive, negative]` triplets from `cache/echo-data/` |
| `Wiki1M` | `Wiki1M` | SimCSE unsupervised contrastive (1M Wikipedia sentences) | `[text, text]` self-pairs from `cache/wiki1m_for_simcse.txt` |
| `Derm1M` | `Derm1M` | Dermatology SimCSE (domain adaptation) | `[text, text]` self-pairs |
| `Derm1M_SimVariants` | `Derm1M_SimVariants` | Dermatology similarity variant pairs (no negatives) | `[query+prompt, positive+prompt]` pairs from JSONL |
| `Derm1M_Variants_Eval` | `Derm1M_Variants_Eval` | Similarity variants with train/val/test splits | `[query, positive, negative]` triplets from split JSONL files |
| `DermVariants` | `DermVariants` | **Main dermatology dataset** — 5 sub-tasks with rich prompt diversity | `[query, positive, negative]` triplets |

**DermVariants sub-tasks** (each with multiple prompt variations randomly sampled per sample):
1. **SemVariants** (10 prompts): Semantic similarity between dermatological descriptions
2. **VisVariants** (5 prompts): Match diagnosis text to visual description text
3. **DermQA** (9 prompts): Dermatology question-answering retrieval
4. **SI1** (9 prompts): Clinical vignette-based dermatology QA
5. **SI2** (7 prompts): Dermatology clinical question retrieval

Instruction placement varies by task: SemVariants/VisVariants/SI1 apply prompt to query, positive, and negative; DermQA/SI2 apply prompt to query only. Supports `dermqa_upsample_ratio` for upsampling.

### Loss: `llm2vec/loss/`

Loss loading is in `llm2vec/loss/utils.py:load_loss()`. Multiple versions exist:

| Version | File | MixCSE | Hard Neg Mining | Mixed Negatives | Focal Reweighting |
|---------|------|--------|-----------------|-----------------|-------------------|
| V0 | `HardNegativeNLLLossV0.py` | No | No | No | No |
| V1 (current) | `HardNegativeNLLLoss.py` | Yes | Yes | Yes (lambda=0.2) | No |
| V2 | `HardNegativeNLLLossV2.py` | Yes | Yes | Yes | Yes (gamma=0.5) |

**Current loss (V1)** — `HardNegativeNLLLoss(nn.Module)`:
1. In-batch positive logits (diagonal labels)
2. Explicit negative logits
3. Hard-negative mining: find the hardest negative per query (highest cosine similarity)
4. Mixed negative: `normalize(lambda * positive + (1-lambda) * hard_neg)`, detached
5. Full logits: `[B, B+N+1]`, standard CrossEntropyLoss
6. DDP support via `mismatched_sizes_all_gather` for gathering across ranks with different batch sizes

### Training Scripts: `experiments/`

Training scripts use HuggingFace `Trainer` with `accelerate` for distributed training and `swanlab` for experiment logging. They read a single JSON config file.

**Evolution of training scripts:**

| Script | Key Feature |
|--------|-----------|
| `run_supervised.py` | Base supervised contrastive training |
| `run_supervised_with_eval.py` | + Retrieval evaluation callback at each checkpoint (BEIR metrics: NDCG, MAP, Recall, Precision) |
| `run_supervised_fusion_withEval.py` | + Layer fusion pooling mode with backbone freezing (imports `llm2vecV3`) |

Common pipeline:
1. Parse JSON config via `HfArgumentParser` with `ModelArguments`, `DataTrainingArguments`, `CustomArguments`, `TrainingArguments`
2. Load model via `LLM2Vec.from_pretrained()` with base model + merged PEFT adapters
3. Apply new LoRA via `initialize_peft()` (targets q/v/k/o/gate/up/down projections), or freeze backbone for fusion-only training
4. Load dataset via `load_dataset()`; load loss via `load_loss()`
5. `LLM2VecSupervisedTrainer(Trainer)` overrides `compute_loss()` (forward query/positive/negative, compute contrastive loss), `get_train_dataloader()` (uses `SequentialSampler` — dataset manages its own ordering)
6. `EvaluateAndLogCallback` (in withEval variants) runs retrieval evaluation on checkpoint saves

**Evaluation scripts:**
- `test_supervised.py` — standalone retrieval evaluation (BEIR metrics) on a trained model
- `encode.py` — batch encode texts to .npy files (supports sharding for distributed encoding)
- `encode_l2v_similarity.py` / `encode_bert_similarity.py` — pairwise cosine similarity computation
- `run_DermSimRetrieval_l2v.py` / `run_DermSimRetrieval_bert.py` — dermatology sentence pair retrieval evaluation
- `benchmark_encode.py` — encoding throughput benchmark (sequences/second)

### Downstream Evaluation: `experiments/src_downstream/`

Two retrieval task types, each implemented for multiple model architectures (LLM2Vec, BERT, GPT-2, ModernBERT, Qwen3-Embedding):

- **Homogeneous Retrieval (`homo_RT_*.py`)**: Given an original dermatology text (query), retrieve the correct positive variant among positive and hard negative variants. Accuracy metric.
- **Non-homogeneous Retrieval (`nonhomo_RT_*.py`)**: Given a dermatology question, retrieve the correct answer among right and wrong choices. BEIR evaluation (NDCG/MAP/Recall/Precision).

Sweep scripts (`Scripts/RT_text/homo_sweep_*.sh`, `Nonhomo_RT_sweep_*.sh`) systematically evaluate multiple training checkpoints.

### BLURB Benchmark: `BLURB-src/`

Biomedical Language Understanding and Reasoning Benchmark integration. Supports running LLM2Vec models on standard biomedical NLP tasks:

- **Sequence Classification** (`seqcls/`): `run_seqcls_llm2vec.py` wraps LLM2Vec as encoder inside `ModelForSeqCls`. Supports single-sentence and sentence-pair classification with 5 fusion modes (concat, mean, max, sub, mul). LoRA fine-tuning or linear probing. Multiple metrics (accuracy, pearsonr, PRF1, HoC, MacroAUROC).
- **Token Classification / NER** (`tokcls/`): `run_ner_llm2vec.py` with `ModelForNER`. Subword merging, label alignment (`retroactive_labels` next_token mode), seqeval evaluation.
- **Question Answering** (`qa/`): Standard extractive QA pipeline.
- **Multiple Choice** (`mc/`): Standard multiple-choice fine-tuning.
- **BLURB Score** (`tab_maker.py`): Computes per-task averages and overall BLURB Score, generates Markdown comparison tables.

BLURB datasets: BC2GM, BC5CDR-chem, BC5CDR-disease, JNLPBA, NCBI-disease (NER); ebmnlp (PICO); chemprot, DDI, GAD (RE); BIOSSES (STS); HoC (DocCls); bioasq, pubmedqa (QA).

### Config Files: `train_configs/` and `test_configs/`

JSON files organized by training stage. Key fields:

**Model specification:**
- `model_name_or_path`: base HF model (e.g., `Meta-Llama-3.1-8B-Instruct`)
- `peft_model_name_or_path`: pre-trained LoRA adapter (MNTP step)
- `extra_model_name_or_path`: list of additional PEFT adapters to chain and merge
- `bidirectional`: enables bidirectional attention
- `pooling_mode`: `"mean"`, `"latent_pooling"`, or `"layer_fusion"`

**Layer fusion (in `Baseline_MixCSE_Fusion/` configs):**
- `layer_fusion_num_layers`, `layer_fusion_temperature`, `layer_fusion_hidden_dim`
- `layer_fusion_train_only`, `layer_fusion_gamma_init`, `layer_fusion_gamma_learnable`

**Dataset:**
- `dataset_name`: maps to class via `load_dataset()` (e.g., `"DermVariants"`, `"E5"`)
- `dataset_file_path`: path to data directory/file
- `dermqa_upsample_ratio`: upsampling factor for DermQA sub-task

**Training hyperparameters:** Standard HF `TrainingArguments` fields plus `lora_r`, `loss_class`, `loss_scale`, `stop_after_n_steps`.

**Evaluation (withEval configs):** `eval_batch_size`, `eval_top_k`, `eval_separator` (`"!@#$%^&*()"`).

### Visualization: `visualization/`

- **UMAP embedding plots** (`embed_l2v_umap.py`, `embed_bert_umap.py`, `embed_gpt_umap.py`): 2D scatter plots of embeddings colored by label, configurable n_neighbors/min_dist
- **kNN Neighborhood Purity** (`knn_l2v.py`, `knn_bert.py`, etc.): Quantitative embedding quality metric — fraction of k nearest neighbors sharing the same label

### Other Files

- `llm2vec/experiment_utils.py`: Functions `generate_experiment_id()` and `parse_experiment_id()` for constructing/parsing experiment ID strings from hyperparameters
- `test_latentpooling_init.py`: Validates loading pre-trained latent attention pooling weights into `LatentAttentionPooling` module
- `examples/`: Demonstrates classification, clustering, retrieval, and STS tasks using LLM2Vec with sklearn or BEIR
- `fetch_eval_metrics.py`: GPU occupancy/utilization tool (allocates GPU memory and maintains target utilization %; filename is misleading)

## Text Separator Convention

The string `!@#$%^&*()` is used as a separator between instruction and document text in tokenization. This is a core convention used in `LLM2Vec.tokenize()` and dataset classes. The `_skip_instruction()` method replaces `attention_mask` with `embed_mask` so pooling only considers content tokens (not instruction tokens).

## Experiment Logging

This project uses [SwanLab](https://swanlab.cn/) for experiment tracking. `SwanLabCallback` is integrated into all training scripts. Logged metrics include training loss, learning rate, and (in withEval scripts) NDCG@{1,3,5,10} at each checkpoint.
