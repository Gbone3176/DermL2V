# DermL2V Inference Weights Manifest

Prepared on 2026-07-18 for the standalone `DermL2V_inference` package.

## Bundled Weights

These adapter/checkpoint weights are copied into this directory and are used by
`infer_derml2v.py` by default when present.

| Local path | Approx. size | Source |
|---|---:|---|
| `weights/llm2vec-mntp` | 169M | `McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp` |
| `weights/llm2vec-mntp-supervised` | 161M | `McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised` |
| `weights/DermL2V_adapter` | 429M | DermL2V adapter release |

The copied Hugging Face snapshots were dereferenced, so `weights/` contains real
files rather than symlinks back into the local cache.

## External Base Model

The Meta-Llama-3.1-8B-Instruct base model is not vendored here. It is about 30G
on this machine and should be obtained separately according to the upstream Meta
Llama license.

For a portable release, either place the base model at:

```text
DermL2V_inference/weights/Meta-Llama-3.1-8B-Instruct
```

or pass it explicitly at runtime:

```bash
python infer_derml2v.py --base_model_name_or_path /path/to/Meta-Llama-3.1-8B-Instruct ...
```
