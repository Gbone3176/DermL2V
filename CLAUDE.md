# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM2Vec converts decoder-only LLMs into text encoders via three steps:
1. Enabling bidirectional attention
2. Masked Next Token Prediction (MNTP) training
3. Unsupervised contrastive learning (SimCSE) or supervised contrastive training

Paper: [LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders](https://arxiv.org/abs/2404.05961)

## Installation

```bash
pip install -e .
pip install flash-attn --no-build-isolation
```

The package requires `transformers>=4.43.1,<=4.44.2`. Training scripts additionally require `accelerate`, `swanlab`, and `peft`.

## Key Commands

### Training

**MNTP (Step 1 — Masked Next Token Prediction):**
```bash
python experiments/run_mntp.py train_configs/mntp/<Model>.json
```

**SimCSE (Step 2 — Unsupervised contrastive):**
```bash
python experiments/run_simcse.py train_configs/simcse/<Model>.json
```

**Supervised contrastive (Step 3 — multi-GPU with torchrun):**
```bash
torchrun --nproc_per_node=<N> experiments/run_supervised.py train_configs/supervised/<Model>.json
```

**Word-level tasks:**
```bash
python -m experiments.run_word_task train_configs/word-task/<Config>.json
python experiments/test_word_task.py --config_file test_configs/word-task/<Config>.json
```

### Evaluation

```bash
pip install llm2vec[evaluation]
python experiments/mteb_eval.py --model_name <HF_MODEL> --task_name <TASK> \
    --task_to_instructions_fp test_configs/mteb/task_to_instructions.json --output_dir results
```

For custom/local models use `experiments/mteb_eval_custom.py` with `--base_model_name_or_path` and `--peft_model_name_or_path`.

## Architecture

### Core class: `llm2vec/llm2vec.py` — `LLM2Vec`

`LLM2Vec` is an `nn.Module` wrapper around HuggingFace models. Key responsibilities:
- **Model loading** via `from_pretrained()`: loads a base model, optionally applies and merges PEFT/LoRA adapters (supports chaining multiple adapters via `extra_model_name_or_path`)
- **Bidirectional model selection**: maps config class names (LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config) to their `*BiModel` counterparts
- **Tokenization**: `prepare_for_tokenization()` wraps text with model-specific instruction templates (e.g., Llama3 chat format, Mistral `[INST]` tags)
- **Encoding**: `encode()` produces embeddings from text inputs in the form `[[instruction, text], ...]` or `[text, ...]`
- **Pooling**: supports `mean`, `weighted_mean`, `eos_token`, `last_token`, `bos_token`, and `latent_pooling` modes
- **Latent attention pooling**: optional `LatentAttentionPooling` module (`llm2vec/pooling_latent.py`) using learnable latent vectors with multihead cross-attention

### Bidirectional models: `llm2vec/models/`

Each supported architecture has a `bidirectional_*.py` file that:
1. Subclasses the attention classes (eager, flash_attention_2, sdpa) and sets `is_causal = False`
2. Subclasses the decoder layer to use the modified attention
3. Provides `*BiModel` (for encoding) and `*BiForMNTP` (for masked next token prediction training)

Supported architectures: Llama, Mistral, Gemma, Qwen2. Adding a new architecture requires creating a new `bidirectional_*.py` following this pattern and registering it in `llm2vec/models/__init__.py` and `LLM2Vec._get_model_class()`.

### Datasets: `llm2vec/dataset/`

All datasets extend `llm2vec.dataset.dataset.Dataset` (a `torch.utils.data.Dataset` subclass) and return `TrainSample` objects containing `texts` and `label`. Key datasets:
- **E5Data**: supervised contrastive data from echo-embeddings (loaded from `cache/echo-data/`)
- **Wiki1M**: 1M Wikipedia sentences for SimCSE (loaded from `cache/wiki1m_for_simcse.txt`)
- **Derm\* variants**: domain-specific dermatology datasets

Dataset loading is centralized in `llm2vec/dataset/utils.py:load_dataset()` which maps string names to classes.

### Loss: `llm2vec/loss/`

`HardNegativeNLLLoss`: MixCSE-style contrastive loss with in-batch positives, optional explicit negatives, hard-negative mining, and mixed negatives. Supports DDP via `mismatched_sizes_all_gather`. Loss loading is in `llm2vec/loss/utils.py:load_loss()`.

### Training scripts: `experiments/`

Training scripts use HuggingFace `Trainer` with `accelerate` for distributed training. They read a single JSON config file that contains both model args and training hyperparameters. The pattern is:
1. Parse JSON config → `HfArgumentParser` with `TrainingArguments`
2. Load model via `LLM2Vec.from_pretrained()`
3. Apply LoRA via `initialize_peft()`
4. Load dataset via `load_dataset()`
5. Custom training loop with `Trainer` subclass

### Config files: `train_configs/` and `test_configs/`

JSON files organized by training stage (`mntp/`, `simcse/`, `supervised/`, `word-task/`). Key fields:
- `model_name_or_path`: base HF model
- `peft_model_name_or_path`: pre-trained LoRA adapter to load
- `dataset_name`: maps to a dataset class via `load_dataset()`
- `dataset_file_path`: path to data files
- `lora_r`, `torch_dtype`, `attn_implementation`: LoRA and model config
- Standard HF `TrainingArguments` fields (`per_device_train_batch_size`, `learning_rate`, etc.)

## Text separator convention

The string `!@#$%^&*()` is used as a separator between instruction and document text in tokenization. This is a core convention used in `LLM2Vec.tokenize()` and dataset classes.
