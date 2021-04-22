#!/bin/sh
#SBATCH -o gpu-job-%j.output 
#SBATCH -p PV100q
#SBATCH --gres=gpu:2 
#SBATCH -n 1

python3 GA_LSTM_test.py snp500 2 6