#!/bin/bash

# Fine-tuning of a speech (unit) BART model
# on ASR task (LS960h) as debugging experiment
# for fine-tuning with different dictionaries

set -eu

fairseq_root="/private/home/jadecopet/projects/gslm/nlp-tasks/fairseq-py"

data_path_list=(
  "/private/home/padentomasello/data/stop/fairseq/post_qa/asr/"
)
name_list=(
  "ls"
)

exp_root_dir="/checkpoint/${USER}/projects/gslm/nlp/bart/finetune-asr-ls-ctc"

#num_gpus=2
#local_flag="--local"

num_gpus=8
local_flag="--constraint volta32gb"

for i in ${!data_path_list[@]}; do
  data_path=${data_path_list[$i]}
  name=${name_list[$i]}
  run_dir="${exp_root_dir}/${name}"
  mkdir -p $run_dir

  echo "#### BART FINE-TUNING: $name"
  echo "output dir: $run_dir"
  NCCL_DEBUG=WARN PYTHON_PATH=$fairseq_root python -m fb_sweep.sweep_ctc_asr \
    --prefix $name \
    --data $data_path \
    --checkpoints-dir=$run_dir \
    --tensorboard-logdir=$run_dir \
    --num-trials -1 \
    --num-gpus $num_gpus \
    --num-nodes 1 \
    --resume-failed \
    --partition learnlab \
    $local_flag
done
