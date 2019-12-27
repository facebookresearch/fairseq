#!/bin/bash
basepath=$(dirname "$0")
for dir in $@;
do
    model_dir=$dir
    num_checkpoints=`ls $model_dir/*.pt | grep -Po "checkpoint[0-9]+" | wc -l`
    mkdir -p $model_dir/slurm-logs
    echo "$model_dir"
    echo "# ckpts: $num_checkpoints"
    sbatch\
        --gres=gpu:1 \
        --partition=learnfair \
        --time=00:20:00 \
        --cpus-per-task 4 \
        -e $model_dir/slurm-logs/infer-valid-%A-%a.err \
        -o $model_dir/slurm-logs/infer-valid-%A-%a.out \
        --array=1-$num_checkpoints $basepath/infer_dev_must_c_en_ro_slurm-array.sh $model_dir $basepath
done
