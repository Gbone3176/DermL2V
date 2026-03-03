# BLURB Commands

## NER with llm2vec (local JSON dataset)

```bash
MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
MODEL_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
task=NCBI-disease_hf
datadir=../data/tokcls/$task
outdir=/storage/BioMedNLP/llm2vec/BLURB-src/runs/$task/$MODEL
mkdir -p $outdir

CUDA_VISIBLE_DEVICES=0 python3 -u BLURB-src/tokcls/run_ner_llm2vec.py --model_name_or_path $MODEL_PATH --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json --do_train --do_eval --do_predict --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --fp16 --learning_rate 5e-5 --warmup_ratio 0.1 --num_train_epochs 4 --max_seq_length 128 --save_strategy no --evaluation_strategy no --output_dir $outdir --overwrite_output_dir --report_to none --merge_subwords True --retroactive_labels same_token --bidirectional True |& tee $outdir/log.txt &
```

## NER with llm2vec (Hugging Face Hub dataset)

```bash
MODEL="YourModelAlias"
MODEL_PATH="/path/to/base/model"
outdir=runs/conll2003/$MODEL
mkdir -p $outdir

CUDA_VISIBLE_DEVICES=0 python3 -u BLURB-src/tokcls/run_ner_llm2vec.py \
  --model_name_or_path $MODEL_PATH \
  --dataset_name conll2003 --dataset_config_name en \
  --do_train --do_eval \
  --per_device_train_batch_size 16 --fp16 \
  --learning_rate 5e-4 --num_train_epochs 3 --max_seq_length 256 \
  --save_strategy no --evaluation_strategy steps --eval_steps 500 --output_dir $outdir --overwrite_output_dir \
  --report_to none --merge_subwords True --retroactive_labels same_token --bidirectional True \
  |& tee $outdir/log.txt &
```

### Notes
Only classifier head is trainable by default; to add LoRA training set `--lora_r > 0`.
JSON/JSONL data must include `tokens` and `ner_tags`.
For strict classifier-only training, do not set `--lora_r`.