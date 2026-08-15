# DermL2V Inference

Standalone inference and RT-full retrieval evaluation for DermL2V. This
repository contains the inference implementation and four LFS-managed test
sets; model weights must be downloaded separately before running the script.

## Preparation

### 1. Clone the repository and fetch the test data

Install [Git LFS](https://git-lfs.com/) before cloning if you plan to run the
RT-full evaluation. The embedding-only example does not use the test data.

```bash
git lfs install
git clone https://github.com/Gbone3176/DermL2V.git
cd DermL2V
git lfs pull
```

### 2. Create a Python environment

Python 3.10 is recommended. For GPU inference, first install a CUDA-compatible
build of PyTorch 2.5.1 for your system using the
[PyTorch installer](https://pytorch.org/get-started/locally/). Then install the
remaining pinned dependencies:

```bash
conda create -n derml2v python=3.10 -y
conda activate derml2v
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available())"
```

An 8B model in fp16 is loaded for CUDA inference, so use a GPU with sufficient
memory. CPU inference is supported but will be substantially slower.

### 3. Download the complete model chain

DermL2V is a sequence of PEFT adapters applied in the following order:

```text
Meta-Llama-3.1-8B-Instruct
  -> LLM2Vec mntp adapter
  -> LLM2Vec supervised adapter
  -> DermL2V adapter with structured self-attention pooling
```

The Llama base model is gated. First accept its license on the
[model page](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct),
then authenticate with an account that has access:

```bash
hf auth login
mkdir -p weights

hf download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir weights/Meta-Llama-3.1-8B-Instruct
hf download McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp \
  --local-dir weights/llm2vec-mntp
hf download McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised \
  --local-dir weights/llm2vec-mntp-supervised
hf download Gbone3176/DermL2V \
  --local-dir weights/DermL2V_adapter
```

After downloading, the relevant directory layout must be:

```text
weights/
├── Meta-Llama-3.1-8B-Instruct/
├── llm2vec-mntp/
├── llm2vec-mntp-supervised/
└── DermL2V_adapter/
    ├── adapter_model.safetensors
    ├── structured_self_attn.pt
    └── llm2vec_config.json
```

`adapter_model.safetensors` is the standard PEFT filename and must not be
renamed. `structured_self_attn.pt` is required for the final pooling layer.
The repository ignores `weights/`, so the downloaded artifacts are never added
to Git by default.

| Component | Source |
|---|---|
| Base model | [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| First adapter | [McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp) |
| Second adapter | [McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised) |
| Final adapter | [Gbone3176/DermL2V](https://huggingface.co/Gbone3176/DermL2V) |

The base model is distributed under the Llama 3.1 Community License. You are
responsible for complying with that license and the policies of each upstream
model repository.

## Encode text

Once the four directories above exist, local defaults resolve automatically:

```bash
python infer_derml2v.py \
  --text "Dermoscopy shows irregular pigment network and blue-white veil."
```

The result is JSON containing the input text and a 4096-dimensional embedding.
Use `--text` repeatedly or provide one non-empty text per line with
`--text_file`.

If weights are stored elsewhere, pass every required path explicitly:

```bash
python infer_derml2v.py \
  --base_model_name_or_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /path/to/llm2vec-mntp \
  --supervised_model_name_or_path /path/to/llm2vec-mntp-supervised \
  --checkpoint_dir /path/to/DermL2V_adapter \
  --text "A new erythematous scaly plaque on sun-exposed skin."
```

The final adapter configuration selects `structured_selfattn` automatically.
Do not override `--pooling_mode` unless intentionally evaluating another
pooling implementation.

## RT-full evaluation

The repository includes four retrieval test sets in `data/`: DermaSynth-E3,
MedMCQA, MedQuAD, and SCE-Derma-SQ. Ensure `git lfs pull` completed before
running evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python infer_derml2v.py \
  --eval_rt_full \
  --batch_size 8
```

Lower `--batch_size` if GPU memory is insufficient. Results are written to
`results/rt_full/DermL2V_rt_full/`, including per-dataset metrics and
`summary_at10.md`. Use `--rt_dataset DermSynth` (or another dataset key) to
evaluate a subset, and `--eval_limit N` only for a quick debugging run.

## Common setup errors

- **"No DermL2V checkpoint found"**: download the final adapter to
  `weights/DermL2V_adapter`, pass `--checkpoint_dir`, or set
  `DERML2V_CHECKPOINT_DIR`.
- **Gated-model access error**: accept the Llama license and run `hf auth login`
  again with an authorized Hugging Face account.
- **Missing or tiny JSONL files**: install Git LFS and run `git lfs pull`.
- **CUDA out of memory**: decrease `--batch_size`; CPU mode is available with
  `--device cpu` but is much slower.

## Repository contents

- `infer_derml2v.py`: command-line inference and RT-full evaluation entry point
- `derml2v_llm2vec/`: vendored LLM2Vec-compatible inference implementation
- `data/`: four Git LFS-managed RT-full test files
- `WEIGHTS_MANIFEST.md`: model-chain provenance and expected directories
- `DATA_MANIFEST.md`: test-set record counts and details

No training code or model weights are included in this GitHub repository.
