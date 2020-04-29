#!/bin/bash
set -e
config=$1
model_dir=$2
model=$model_dir/checkpoint$SLURM_ARRAY_TASK_ID.pt
echo $model
./scripts/fast-infer.sh $config $model
