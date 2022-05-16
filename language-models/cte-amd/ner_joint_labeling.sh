#!/bin/bash
# Script to run joint labelling experiments

MODELS='/gpfs/projects/bsc88/projects/bio_eval/power9/models/mbert-cased /gpfs/projects/bsc88/projects/bne/eval_amd/models/beto /gpfs/projects/bsc88/projects/bio_eval/power9/models/bne-base /gpfs/projects/bsc88/projects/bio_eval/power9/models/BETO_Galen_ckpt /gpfs/projects/bsc88/projects/bio_eval/power9/models/mBERT_Galen_ckpt'
for seed in 1 2 3; do
for learning_rate in 5e-5 1e-5 5e-6 1e-6; do
for model_name in $MODELS; do
for dataset_name in wl clinical_trials; do
  echo "running exp $seed $learning_rate $model_name $dataset_name $gradient_accumulation"
  sbatch /gpfs/projects/bsc88/temutauro/ccasimiro/clinical-nested-ner/language-models/cte-amd/ner.sh $seed $model_name $dataset_name $learning_rate
done
done
done
done
