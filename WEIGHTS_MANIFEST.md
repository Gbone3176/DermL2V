# DermL2V Inference Weights Manifest

The GitHub repository contains no model weights. Download the following model
chain into the listed local directories before running `infer_derml2v.py`.

| Local path | Source | Role |
|---|---|---|
| `weights/Meta-Llama-3.1-8B-Instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Gated Llama base model |
| `weights/llm2vec-mntp` | `McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp` | First LLM2Vec adapter |
| `weights/llm2vec-mntp-supervised` | `McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised` | Second LLM2Vec adapter |
| `weights/DermL2V_adapter` | `Gbone3176/DermL2V` | Final DermL2V adapter and structured self-attention parameters |

The base model is gated and subject to the Llama 3.1 Community License. Accept
the upstream license and authenticate with `hf auth login` before downloading.

The final adapter directory must include both `adapter_model.safetensors` and
`structured_self_attn.pt`. The first is the standard PEFT adapter filename; do
not rename it. See the [README](README.md#preparation) for environment setup
and exact download commands.
