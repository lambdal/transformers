#!/bin/bash
NUM_GPU=${1:-1}
SQUAD_DIR=${2:-"/home/ubuntu/transformers/data"}
LOG_DIR=${3:-"/home/ubuntu/transformers/log_2080Tix1"}

mkdir -p ${LOG_DIR}

# FP32 ---------------------------------------------------------------
# BERT base uncased
if [ -d "output/squad_bert_base_uncased" ]; then rm -Rf "output/squad_bert_base_uncased"; fi
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU ./examples/run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir output/squad_bert_base_uncased/ \
  --per_gpu_train_batch_size=12 \
  |& tee ${LOG_DIR}/squad_bert_base_uncased.txt


# BERT large uncased whole word masking
if [ -d "output/squad_bert_large_uncased_whole_word_masking" ]; then rm -Rf "output/squad_bert_large_uncased_whole_word_masking"; fi
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU ./examples/run_squad.py \
 --model_type bert \
 --model_name_or_path bert-large-uncased-whole-word-masking \
 --do_train \
 --do_eval \
 --do_lower_case \
 --train_file $SQUAD_DIR/train-v1.1.json \
 --predict_file $SQUAD_DIR/dev-v1.1.json \
 --learning_rate 3e-5 \
 --num_train_epochs 2 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir output/squad_bert_large_uncased_whole_word_masking/ \
 --per_gpu_eval_batch_size=3 \
 --per_gpu_train_batch_size=3 \
 |& tee ${LOG_DIR}/squad_bert_large_uncased_whole_word_masking.txt



# DistilBert
if [ -d "output/squad_distilbert-base-uncased" ]; then rm -Rf "output/squad_distilbert-base-uncased"; fi
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU ./examples/run_squad.py \
 --model_type distilbert \
 --model_name_or_path distilbert-base-uncased \
 --do_train \
 --do_eval \
 --do_lower_case \
 --train_file $SQUAD_DIR/train-v1.1.json \
 --predict_file $SQUAD_DIR/dev-v1.1.json \
 --per_gpu_train_batch_size 12 \
 --learning_rate 3e-5 \
 --num_train_epochs 2.0 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir output/squad_distilbert-base-uncased \
 |& tee ${LOG_DIR}/squad_distilbert-base-uncased.txt




# FP16 ---------------------------------------------------------------
# BERT base uncased
if [ -d "output/squad_bert_base_uncased_fp16" ]; then rm -Rf "output/squad_bert_base_uncased_fp16"; fi
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU ./examples/run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir output/squad_bert_base_uncased_fp16/ \
  --per_gpu_train_batch_size=12 \
  --fp16 \
  |& tee ${LOG_DIR}/squad_bert_base_uncased_fp16.txt


# BERT large uncased whole word masking
if [ -d "output/squad_bert_large_uncased_whole_word_masking_fp16" ]; then rm -Rf "output/squad_bert_large_uncased_whole_word_masking_fp16"; fi
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU ./examples/run_squad.py \
 --model_type bert \
 --model_name_or_path bert-large-uncased-whole-word-masking \
 --do_train \
 --do_eval \
 --do_lower_case \
 --train_file $SQUAD_DIR/train-v1.1.json \
 --predict_file $SQUAD_DIR/dev-v1.1.json \
 --learning_rate 3e-5 \
 --num_train_epochs 2 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir output/squad_bert_large_uncased_whole_word_masking_fp16/ \
 --per_gpu_eval_batch_size=3 \
 --per_gpu_train_batch_size=3 \
 --fp16 \
 |& tee ${LOG_DIR}/squad_bert_large_uncased_whole_word_masking_fp16.txt



# DistilBert
if [ -d "output/squad_distilbert-base-uncased_fp16" ]; then rm -Rf "output/squad_distilbert-base-uncased_fp16"; fi
python -m torch.distributed.launch --nproc_per_node=$NUM_GPU ./examples/run_squad.py \
 --model_type distilbert \
 --model_name_or_path distilbert-base-uncased \
 --do_train \
 --do_eval \
 --do_lower_case \
 --train_file $SQUAD_DIR/train-v1.1.json \
 --predict_file $SQUAD_DIR/dev-v1.1.json \
 --per_gpu_train_batch_size 12 \
 --learning_rate 3e-5 \
 --num_train_epochs 2.0 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir output/squad_distilbert-base-uncased_fp16 \
 --fp16 \
 |& tee ${LOG_DIR}/squad_distilbert-base-uncased_fp16.txt


