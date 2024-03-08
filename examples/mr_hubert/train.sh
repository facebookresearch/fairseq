#!/bin/bash

FAIRSEQ=  # Setup your fairseq directory

config_dir=${FAIRSEQ}/examples/mr_hubert/config
config_name=mr_hubert_base_librispeech

# Prepared Data Directory
data_dir=librispeech
# -- data_dir
#    -- train.tsv
#    -- valid.tsv

label_dir=labels
# -- label_dir
#    -- train.km
#    -- valid.km
#    -- dict.km.txt


exp_dir=exp     # Target experiments directory
ratios="[1, 2]" # Default label rate ratios
label_rate=50   # Base label rate


_opts=

# If use slurm, uncomment this line and modify the job submission at
# _opts="${_opts} hydra/launcher=submitit_slurm +hydra.launcher.partition=${your_slurm_partition} +run=submitit_reg"

# If want to set additional experiment tag, uncomment this line
# _opts="${_opts} hydra.sweep.subdir=${your_experiment_tag}"


python ${FAIRSEQ}/fairseq_cli/hydra_train.py \
  -m --config-dir ${config_dir} --config-name ${config_name} ${_opts} \
  task.data=${data_dir} \
  task.label_dir=${label_dir} \
  task.labels='["km"]' \
  model.label_rate=${label_rate} \
  task.label_rate_ratios='${ratios}' \
  hydra.sweep.dir=${exp_dir} &



