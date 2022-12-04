#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=wav2vec,learnlab,learnfair 
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/%u/jupyter-%j.log

conda activate ${CONDA_ENV}
cat /etc/hosts
jupyter-lab --ip=0.0.0.0 --port=${1:-8888} # use your desired port 