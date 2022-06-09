#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES=0

source $SCRIPT_DIR/../venv/bin/activate

model_name=$1
dataset_name=$2
seed=$3

dt=$(date +%m-%d-%Y-%T)
run_dir="$SCRIPT_DIR/runs/$model_name/$dataset_name/$seed/$dt"
mkdir -p $run_dir

loading_script="$SCRIPT_DIR/../datasets/load_ner_iob2.py"
python $SCRIPT_DIR/run_ner.py \
  --model_name_or_path $model_name \
  --loading_script $loading_script \
  --dataset_name $dataset_name \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 20 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --overwrite_output_dir \
  --seed $seed \
  --logging_dir $run_dir/tb \
  --output_dir $run_dir 2>&1 | tee $run_dir/train.log
