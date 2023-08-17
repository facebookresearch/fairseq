#!/usr/bin/env zsh

job_id=$1
task_id=$2
dir="$3"
cp="$dir/checkpoints/checkpoint_last.pt"

echo "job_id: $job_id, task_id: $task_id, dir: $dir"

declare -A tasks
tasks[cola]="/fsx-wav2vec/abaevski/data/nlp/GLUE/CoLA-bin"
tasks[qnli]="/fsx-wav2vec/abaevski/data/nlp/GLUE/QNLI-bin"
tasks[mrpc]="/fsx-wav2vec/abaevski/data/nlp/GLUE/MRPC-bin"
tasks[rte]="/fsx-wav2vec/abaevski/data/nlp/GLUE/RTE-bin"
tasks[sst_2]="/fsx-wav2vec/abaevski/data/nlp/GLUE/SST-2-bin"

for task data_path in ${(kv)tasks}; do
    for lr in 5e-6 8e-6 1e-5 2e-5 5e-5 8e-5 1e-4 2e-4; do
      PYTHONPATH=. PREFIX="${PREFIX}" SUFFIX="" nohup python fairseq_cli/hydra_train.py -m --config-dir examples/roberta/config/finetuning \
      --config-name $task hydra/launcher=submitit_slurm +run_config=slurm_1g_aws task.data="$data_path" hydra.launcher.name=finetune_${task}_${PREFIX} \
      checkpoint.restore_file="$cp" +hydra.launcher.additional_parameters.dependency="afterok:$job_id" hydra.sweep.dir="$dir/finetune_lr/$task/$lr" "optimization.lr=[${lr}]" &
    done
done
