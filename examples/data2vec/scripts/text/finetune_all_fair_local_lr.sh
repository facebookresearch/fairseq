#!/usr/bin/env zsh

dir="$1"
cp="$dir/checkpoints/checkpoint_last.pt"

echo "dir: $dir"

declare -A tasks
tasks[cola]="/fsx-wav2vec/abaevski/data/nlp/GLUE/CoLA-bin"
tasks[qnli]="/fsx-wav2vec/abaevski/data/nlp/GLUE/QNLI-bin"
tasks[mrpc]="/fsx-wav2vec/abaevski/data/nlp/GLUE/MRPC-bin"
tasks[rte]="/fsx-wav2vec/abaevski/data/nlp/GLUE/RTE-bin"
tasks[sst_2]="/fsx-wav2vec/abaevski/data/nlp/GLUE/SST-2-bin"

lrs=(5e-6 8e-6 1e-5 2e-5)

for task data_path in ${(kv)tasks}; do
    for lr in $lrs; do
      echo $lr $task
      PYTHONPATH=. PREFIX="${PREFIX}" SUFFIX="" \
        python fairseq_cli/hydra_train.py -m --config-dir examples/roberta/config/finetuning \
        --config-name $task +run_config=local task.data="$data_path" common.log_interval=200 dataset.num_workers=1 \
        checkpoint.restore_file="$cp" hydra.sweep.dir="$dir/finetune_lr/$task/$lr" "optimization.lr=[${lr}]"
    done
done
