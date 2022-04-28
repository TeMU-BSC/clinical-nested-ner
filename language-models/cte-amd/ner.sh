#!/bin/bash
#SBATCH --job-name="ner_joint_labeling"
#SBATCH -D .
#SBATCH --output=logs/ner_joint_labeling_%j.out
#SBATCH --error=logs/ner_joint_labeling_%j.err
#SBATCH --ntasks=1
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task=128
#SBATCH --time=2-0:00:00

module load mkl/2018.4 gcc/10.2.0 rocm/4.0.1 intel/2018.4 python/3.7.4 

seed=$1
model_name=$2
dataset_name=$3

SCRIPT_DIR=/gpfs/projects/bsc88/temutauro/ccasimiro/clinical-nested-ner/language-models/cte-amd
source /gpfs/projects/bsc88/projects/bio_eval/env/bin/activate

export LD_LIBRARY_PATH=/gpfs/projects/bsc88/projects/bne/eval_amd/scripts_to_run/external-lib:$LD_LIBRARY_PATH

seed=$1
model_name=$2
dataset_name=$3
learning_rate=$4

gradient_accumulation=2
epochs=20
batch_size=8

# model_name=/gpfs/projects/bsc88/projects/bio_eval/power9/models/biomedical-clinical-swm-vocab-50k
loading_script="$SCRIPT_DIR/load_ner_iob2.py"
# dataset_name="wl"

# dt=$(date +%m-%d-%Y-%T)
run_dir="$SCRIPT_DIR/runs/$(basename $model_name)/$dataset_name/seed-$seed/lr-$learning_rate"
if [[ ! -d "$run_dir" ]] && [[ ! -f "$run_dir/predict_results.json" ]] ; then
mkdir -p $run_dir

  python $SCRIPT_DIR/run_ner.py \
    --model_name_or_path $model_name \
    --loading_script $loading_script \
    --dataset_name $dataset_name \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation \
    --learning_rate $learning_rate \
    --num_train_epochs $epochs \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --overwrite_output_dir \
    --seed $seed \
    --logging_dir $run_dir/tb \
    --cache_dir $run_dir \
    --output_dir $run_dir 2>&1 | tee $run_dir/train.log

  rm -r $run_dir/checkpoint*

fi

