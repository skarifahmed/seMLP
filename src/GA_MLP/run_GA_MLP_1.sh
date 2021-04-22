#!/bin/sh
#SBATCH -o gpu-job-%j.output 
#SBATCH -p NV100q
#SBATCH --gres=gpu:2 
#SBATCH -n 1

python3 GA_MLP_1.py snp500 2 6