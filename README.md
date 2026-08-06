# DermL2V Inference

Minimal inference entry point for loading the full DermL2V adapter chain and
encoding text into an embedding.

The default path configuration targets the local DermL2V Llama-3.1-8B setup:

- base model: Meta-Llama-3.1-8B-Instruct
- first adapter: `weights/llm2vec-mntp`
- second adapter: `weights/llm2vec-mntp-supervised`
- final adapter: `weights/DermL2V_adapter`

The base model and adapters are external artifacts. Download the gated base
model and the two public LLM2Vec adapters separately, then pass their local
paths to the inference script. The final DermL2V adapter is available from the
companion Hugging Face repository.

Bundled DermL2V checkpoint:

```text
DermL2V_inference/weights/DermL2V_adapter
```

## Usage

From the repository root:

```bash
python DermL2V_inference/infer_derml2v.py \
  --base_model_name_or_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /path/to/llm2vec-mntp \
  --supervised_model_name_or_path /path/to/llm2vec-mntp-supervised \
  --checkpoint_dir /path/to/DermL2V_adapter \
  --text "Dermoscopy shows irregular pigment network and blue-white veil."
```

For a different checkpoint:

```bash
python DermL2V_inference/infer_derml2v.py \
  --checkpoint_dir /path/to/DermL2V_adapter \
  --text "A new erythematous scaly plaque on sun-exposed skin."
```

The output is JSON with `dim`, `count`, and one embedding vector per input text.
The default 8B DermL2V line produces 4096-dimensional embeddings; `b-2048` in
the checkpoint path is the training batch size, not the embedding dimension.
The pooling configuration is read from the checkpoint `llm2vec_config.json` by
default, so this checkpoint loads `structured_selfattn` rather than the old
mean-pooling baseline.

## RT-Full Test Sets

Run the four default nonhomogeneous retrieval test sets with the local `l2v`
environment:

```bash
CUDA_VISIBLE_DEVICES=4 python DermL2V_inference/infer_derml2v.py \
  --base_model_name_or_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --peft_model_name_or_path /path/to/llm2vec-mntp \
  --supervised_model_name_or_path /path/to/llm2vec-mntp-supervised \
  --checkpoint_dir /path/to/DermL2V_adapter \
  --eval_rt_full \
  --batch_size 8
```

The default output directory is:

```text
DermL2V_inference/results/rt_full/DermL2V_rt_full
```

The RT mode loads the model once, encodes query and document sides for each
dataset, and writes per-dataset metric JSON files plus `summary_at10.md`.

The four RT-full JSONL files are vendored under `data/` and named with the
dataset abbreviations from `local_info/local_path.md`:

- `data/DermaSynth-E3.jsonl`
- `data/MedMCQA.jsonl`
- `data/MedQuAD.jsonl`
- `data/SCE-Derma-SQ.jsonl`

See `DATA_MANIFEST.md` for source paths and record counts.

## Weights

See `WEIGHTS_MANIFEST.md` for copied adapter/checkpoint provenance and the base
model requirement.

For a local bundle, large weight files can be tracked with Git LFS. The local
`.gitattributes` marks:

- `weights/**/*.safetensors`
- `weights/**/*.pt`
- `weights/**/*.bin`

## Standalone Repository Notes

This directory vendors the DermL2V-compatible LLM2Vec inference code under
`derml2v_llm2vec/`, so the Python code path does not depend on files outside
this directory. The only required external model artifact is the Llama base
model, unless it is placed under `weights/Meta-Llama-3.1-8B-Instruct`.

No training code is required for this entry point.
