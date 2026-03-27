# #--------------------tokcls--------------------#

# # # ===== Debug: NER (BLURB NCBI-disease llm2vec) =====
# #################################1111111111111#############################################
# # BC2GM_hf, BC5CDR-chem_hf, BC5CDR-disease_hf**, ebmnlp_hf, JNLPBA_hf, NCBI-disease_hf**  #
# ###########################################################################################

TASK="BC2GM_hf"
MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
MODEL_PATH="${MODEL_PATH:-/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct}"
datadir="/storage/LinkBERT/data/seqcls/${TASK}"
outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
mkdir -p "$outdir"

CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.tokcls.run_ner_llm2vec \
  --model_name_or_path "${MODEL_PATH}" \
  --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
  --train_file "${datadir}/train.json" \
  --validation_file "${datadir}/dev.json" \
  --test_file "${datadir}/test.json" \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --fp16 \
  --learning_rate 5e-4 \
  --warmup_ratio 0.5 \
  --num_train_epochs 30 \
  --max_seq_length 512 \
  --save_strategy no \
  --evaluation_strategy no \
  --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
  --overwrite_output_dir \
  --report_to none \
  --merge_subwords True \
  --retroactive_labels next_token \
  --bidirectional True

echo "==== NER $TASK finished ===="


# TASK="BC5CDR-chem_hf"
# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.tokcls.run_ner_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --per_device_train_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 20 \
#   --max_seq_length 512 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --merge_subwords True \
#   --retroactive_labels next_token \
#   --bidirectional True

# echo "==== NER $TASK finished ===="


# TASK="BC5CDR-disease_hf"
# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.tokcls.run_ner_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --per_device_train_batch_size 16 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 20 \
#   --max_seq_length 512 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --merge_subwords True \
#   --retroactive_labels next_token \
#   --bidirectional True

# echo "==== NER $TASK finished ===="


# TASK="ebmnlp_hf"
# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.tokcls.run_ner_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 2 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 1 \
#   --max_seq_length 512 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --merge_subwords True \
#   --retroactive_labels next_token \
#   --bidirectional True

# echo "==== NER $TASK finished ===="


# TASK="JNLPBA_hf"
# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.tokcls.run_ner_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --per_device_train_batch_size 16 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 20 \
#   --max_seq_length 512 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --merge_subwords True \
#   --retroactive_labels next_token \
#   --bidirectional True

# echo "==== NER $TASK finished ===="


# TASK="NCBI-disease_hf"
# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.tokcls.run_ner_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 2 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 1 \
#   --max_seq_length 512 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --merge_subwords True \
#   --retroactive_labels next_token \
#   --bidirectional True

# echo "==== NER $TASK finished ===="


# #--------------------seqcls--------------------#
# ##############################################################################
# # 2 sentences: bioasq_hf, BIOSSES_hf(pearsonr)**, pubmedqa_hf**                  #
# # 1 sentence: chemprot_hf(PRF1), DDI_hf(PRF1)**, GAD_hf(PRF1), HoC_hf(hoc)**     #
# ##############################################################################

# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# TASK="bioasq_hf"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# ############### ===== Debug: SeqCls-2sts (BLURB pubmedqa_hf llm2vec) =====

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.seqcls.run_seqcls_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 40 \
#   --max_seq_length 512 \
#   --pooling_mode latent_pooling \
#   --classifier_dropout 0.1 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --bidirectional True \
#   --pair_fusion concat \

# echo "==== SeqCls $TASK finished ===="


# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# TASK="BIOSSES_hf"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# ############### ===== Debug: SeqCls-2sts (BLURB pubmedqa_hf llm2vec) =====

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.seqcls.run_seqcls_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --per_device_train_batch_size 16 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 20 \
#   --max_seq_length 512 \
#   --pooling_mode latent_pooling \
#   --classifier_dropout 0.1 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --pair_fusion concat \
#   --bidirectional True \
#   --metric_name pearsonr

# echo "==== SeqCls $TASK finished ===="


# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# TASK="pubmedqa_hf"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# ############### ===== Debug: SeqCls-2sts (BLURB pubmedqa_hf llm2vec) =====

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.seqcls.run_seqcls_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --per_device_train_batch_size 16 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 20 \
#   --max_seq_length 512 \
#   --pooling_mode latent_pooling \
#   --classifier_dropout 0.1 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --bidirectional True \
#   --pair_fusion concat \

# echo "==== SeqCls $TASK finished ===="


# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# TASK="chemprot_hf"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# ############### ===== Debug: SeqCls-2sts (BLURB pubmedqa_hf llm2vec) =====

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.seqcls.run_seqcls_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --per_device_train_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-5 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 5 \
#   --max_seq_length 512 \
#   --pooling_mode latent_pooling \
#   --classifier_dropout 0.1 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --bidirectional True \
#   --metric_name PRF1
  
# echo "==== SeqCls $TASK finished ===="


# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# TASK="DDI_hf"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# ############### ===== Debug: SeqCls-2sts (BLURB pubmedqa_hf llm2vec) =====

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.seqcls.run_seqcls_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --per_device_train_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 5 \
#   --max_seq_length 512 \
#   --pooling_mode latent_pooling \
#   --classifier_dropout 0.1 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --bidirectional True \
#   --metric_name PRF1

# echo "==== SeqCls $TASK finished ===="


# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# TASK="GAD_hf"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# ############### ===== Debug: SeqCls-2sts (BLURB pubmedqa_hf llm2vec) =====

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.seqcls.run_seqcls_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --per_device_train_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 10 \
#   --max_seq_length 512 \
#   --pooling_mode latent_pooling \
#   --classifier_dropout 0.1 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --bidirectional True \
#   --metric_name PRF1

# echo "==== SeqCls $TASK finished ===="


# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# TASK="HoC_hf"
# datadir="/storage/LinkBERT/data/seqcls/${TASK}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# ############### ===== Debug: SeqCls-2sts (BLURB pubmedqa_hf llm2vec) =====

# CUDA_VISIBLE_DEVICES=0 python3 -m BLURB-src.seqcls.run_seqcls_llm2vec \
#   --model_name_or_path "/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --per_device_train_batch_size 32 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-4 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 20 \
#   --max_seq_length 512 \
#   --pooling_mode latent_pooling \
#   --classifier_dropout 0.1 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${TASK}" \
#   --overwrite_output_dir \
#   --report_to none \
#   --bidirectional True \
#   --metric_name hoc

# echo "==== SeqCls $TASK finished ===="
