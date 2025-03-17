#!/bin/bash

#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=32                   # Number of CPU cores per task
#SBATCH --mem=64gb                          # Job memory request
#SBATCH --time=96:00:00                     # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
# -------------------------------

srun python -u scripts_bsl/data_stat_bsl.py 