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

for task data_path in ${(kv)tasks}; do
    PYTHONPATH=. PREFIX="${PREFIX}" SUFFIX="" nohup python fairseq_cli/hydra_train.py -m --config-dir examples/roberta/config/finetuning \
    --config-name $task hydra/launcher=submitit_slurm +run_config=slurm_1g_aws task.data="$data_path" hydra.launcher.name=finetune_${task}_${PREFIX} \
    checkpoint.restore_file="$cp" hydra.sweep.dir="$dir/finetune/$task" &
done
