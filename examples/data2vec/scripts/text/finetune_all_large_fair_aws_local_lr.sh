#!/bin/bash

set -eu

job_id="$1"
task_id="$2"
dir="$3"

echo "job_id: $job_id, task_id: $task_id, dir: $dir"

mkdir -p "$dir/log"
sbatch_args="-p wav2vec --nodes=1 --ntasks-per-node=1"
sbatch_args="$sbatch_args --gpus-per-node=1 --cpus-per-task=8 --mem=0 --time=24:00:00"
sbatch_args="$sbatch_args -d afterok:$job_id -o $dir/log/decode_sweep_%A.out"
sbatch_args="$sbatch_args -e $dir/log/decode_sweep_%A.err"

sbatch $sbatch_args examples/data2vec/scripts/text/finetune_all_large_fair_local_lr.sh $dir
