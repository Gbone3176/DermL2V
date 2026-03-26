#--------------------tokcls--------------------#

# # ===== Debug: NER (BLURB NCBI-disease llm2vec) =====
#################################1111111111111#############################################
# BC2GM_hf, BC5CDR-chem_hf, BC5CDR-disease_hf, ebmnlp_hf, JNLPBA_hf, NCBI-disease_hf      #
###########################################################################################

MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
MODEL_PATH="${MODEL_PATH:-/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct}"
task="BC2GM_hf"
datadir="/storage/LinkBERT/data/seqcls/${task}"
outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${task}"
PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
mkdir -p "$outdir"

CUDA_VISIBLE_DEVICES=5 python3 -m BLURB-src.tokcls.run_ner_llm2vec \
  --model_name_or_path "${MODEL_PATH}" \
  --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
  --train_file "/storage/LinkBERT/data/tokcls/NCBI-disease_hf/train.json" \
  --validation_file "/storage/LinkBERT/data/tokcls/NCBI-disease_hf/dev.json" \
  --test_file "/storage/LinkBERT/data/tokcls/NCBI-disease_hf/test.json" \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --fp16 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --num_train_epochs 4 \
  --max_seq_length 4 \
  --save_strategy no \
  --evaluation_strategy no \
  --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/NCBI-disease_hf/mntp-simcseMeta-Llama-3.1-8B-Instruct-debug" \
  --overwrite_output_dir \
  --report_to none \
  --merge_subwords True \
  --retroactive_labels next_token \
  --bidirectional True

echo "==== NER (NCBI-disease) finished ===="




#--------------------seqcls--------------------#
##############################################################################
# 2 sentences: bioasq_hf, BIOSSES_hf(pearsonr), pubmedqa_hf                  #
# 1 sentence: chemprot_hf(PRF1), DDI_hf(PRF1), GAD_hf(PRF1), HoC_hf(hoc)     #
##############################################################################

# MODEL="mntp-simcseMeta-Llama-3.1-8B-Instruct-debug"
# MODEL_PATH="${MODEL_PATH:-/cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct}"
# task="GAD_hf"
# datadir="/storage/LinkBERT/data/seqcls/${task}"
# outdir="/storage/BioMedNLP/llm2vec/BLURB-src/outputs/${MODEL}/${task}"
# PEFT_PATH="/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9"
# mkdir -p "$outdir"

# ############### ===== Debug: SeqCls-2sts (BLURB pubmedqa_hf llm2vec) =====

# CUDA_VISIBLE_DEVICES=5 python3 -m BLURB-src.seqcls.run_seqcls_llm2vec \
#   --model_name_or_path "${MODEL_PATH}" \
#   --peft_addr "/storage/BioMedNLP/llm2vec/output/mntp-simcse/Meta-Llama-3.1-8B-Instruct-debug/checkpoint-9" \
#   --do_train \
#   --do_eval \
#   --do_predict \
#   --train_file "${datadir}/train.json" \
#   --validation_file "${datadir}/dev.json" \
#   --test_file "${datadir}/test.json" \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 1 \
#   --fp16 \
#   --learning_rate 5e-5 \
#   --warmup_ratio 0.1 \
#   --num_train_epochs 2 \
#   --max_seq_length 4 \
#   --pooling_mode latent_pooling \
#   --classifier_dropout 0.1 \
#   --save_strategy no \
#   --evaluation_strategy no \
#   --output_dir "/storage/BioMedNLP/llm2vec/BLURB-src/outputs/mntp-simcseMeta-Llama-3.1-8B-Instruct-debug/pubmedqa_hf" \
#   --overwrite_output_dir \
#   --report_to none \
#   --metric_name PRF1

# echo "==== SeqCls $task finished ===="
