#!/bin/bash

#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=280gb                          # Job memory request
#SBATCH --time=96:00:00                     # Time limit hrs:min:sec
###SBATCH --partition=gpu
###SBATCH --partition=ddp-4way       
###SBATCH --gres=gpu:1              
###SBATCH --constraint=[a40|a6000|h100|rtx8000]
# -------------------------------

# srun -u python scripts_bsl/prepare_training.py --num-workers=64 --split=train --chunksize=1000 --max-num=1000000
# srun -u python scripts_bsl/data_stat_bsl.py
# srun -u python locallaunch.py projects/retri/signclip_bsl/bobsl_islr_finetune.yaml --jobtype local_single
# srun -u python locallaunch.py projects/retri/signclip_bsl/test_bobsl_islr_finetune.yaml --jobtype local_predict 
srun -u python scripts_bsl/test_recognition_supervised.py