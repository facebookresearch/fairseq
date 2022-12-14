#!/usr/bin/env zsh

dir="$1"
cp="$dir/checkpoints/checkpoint_last.pt"

echo "dir: $dir"

declare -A tasks
tasks[qnli]="/private/home/jgu/data/GLUE/QNLI-bin"
tasks[sst_2]="/private/home/jgu/data/GLUE/SST-2-bin"

lrs="5e-6 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4 1e-3"

for task data_path in ${(kv)tasks}; do
  for lr in $(echo "$lrs"); do
    PYTHONPATH=. PREFIX="${PREFIX}" SUFFIX="" nohup python fairseq_cli/hydra_train.py -m --config-dir examples/roberta/config/finetuning \
    --config-name $task hydra/launcher=submitit_slurm +run_config=slurm_1g task.data="$data_path" hydra.launcher.name=finetune_${task}_${PREFIX} \
    checkpoint.restore_file="$cp" hydra.sweep.dir="$dir/finetune_sweep/$task/lr_$lr" "optimization.lr=[${lr}]" &
  done
done
