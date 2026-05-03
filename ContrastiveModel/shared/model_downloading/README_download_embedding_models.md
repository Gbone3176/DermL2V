# Download Embedding Model Snapshots

Use `download_embedding_models.py` to download the five target embedding/retrieval models for migration to another machine.

Default target models:

- `NeuML/pubmedbert-base-embeddings`
- `emilyalsentzer/Bio_ClinicalBERT`
- `BMRetriever/BMRetriever-7B`
- `Qwen/Qwen3-Embedding-8B`
- `nvidia/NV-Embed-v2`

Full snapshot download:

```bash
conda run -n qwen3 python share/download_embedding_models.py \
  --output-dir /cache/hf_model_snapshots/embedding_models
```

Dry-run:

```bash
conda run -n qwen3 python share/download_embedding_models.py --dry-run
```

Download only remote code/config/tokenizer files:

```bash
conda run -n qwen3 python share/download_embedding_models.py --mode code-only
```

Download selected models:

```bash
conda run -n qwen3 python share/download_embedding_models.py \
  --models pubmedbert bioclinicalbert nvembedv2
```

For gated or rate-limited downloads, pass `--token` or set `HF_TOKEN`.
`nvidia/NV-Embed-v2` should be loaded with `trust_remote_code=True` after download.
