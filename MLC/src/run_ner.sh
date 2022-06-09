#!/bin/bash
#---------------Script SBATCH - NLHPC ----------------
#SBATCH -J ner
#SBATCH -p gpus
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -c 1
#SBATCH --mem-per-cpu=4363
#SBATCH --mail-user=matirojasga@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -o ner_%j.out
#SBATCH -e ner_%j.err
ml CUDA/10.2.89
ml Python/3.7.3
/home/mrojasg/clinical-nested-ner/venv/bin/python main.py