#!/bin/bash

FAIRSEQ=  # Setup your fairseq directory

config_dir=${FAIRSEQ}/examples/mr_hubert/config
config_name=mr_hubert_base_librispeech

# override configs if need
max_tokens=3200000
max_sample_size=1000000
max_update=50000


# Prepared Data Directory

data_dir=librispeech
# -- data_dir
#    -- train.tsv
#    -- train.ltr
#    -- valid.tsv
#    -- valid.ltr
#    -- dict.ltr.txt


exp_dir=exp     # Target experiments directory
ratios="[1, 2]" # Default label rate ratios
hubert_path=/path/of/your/hubert.pt

_opts=

# If use slurm, uncomment this line and modify the job submission at
# _opts="${_opts} hydra/launcher=submitit_slurm +hydra.launcher.partition=${your_slurm_partition} +run=submitit_reg"

# If want to set additional experiment tag, uncomment this line
# _opts="${_opts} hydra.sweep.subdir=${your_experiment_tag}"


python ${FAIRSEQ}/fairseq_cli/hydra_train.py \
  -m --config-dir ${config_dir} --config-name ${config_name} ${_opts} \
  task.data=${data_dir} +task.max_sample_size=${max_sample_size} \
  task.label_dir=${data_dir} \
  task.label_rate_ratios='${ratios}' \
  dataset.max_tokens=${max_tokens} \
  optimization.max_update=${max_update} \
  model.multires_hubert_path=${hubert_path} \
  hydra.sweep.dir=${exp_dir} &
