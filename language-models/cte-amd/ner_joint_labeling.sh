#!/bin/bash
# Script to run joint labelling experiments

for seed in 1 2 3; do
for learning_rate in 5e-5 1e-5 5e-6 1e-6; do
for model_name in /gpfs/projects/bsc88/projects/bio_eval/power9/models/biomedical-clinical-swm-vocab-50k /gpfs/projects/bsc88/projects/bio_eval/power9/models/biomedical-swm-vocab-50k; do
for dataset_name in wl pharmaconer clinical_trials; do
  echo "running exp $seed $learning_rate $model_name $dataset_name $gradient_accumulation"
  sbatch /gpfs/projects/bsc88/temutauro/ccasimiro/clinical-nested-ner/language-models/cte-amd/ner.sh $seed $model_name $dataset_name $learning_rate
done
done
done
done
