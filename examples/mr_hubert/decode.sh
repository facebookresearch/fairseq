#!/bin/bash

FAIRSEQ=  # Setup your fairseq directory

config_dir=${FAIRSEQ}/examples/mr_hubert/config
config_name=mr_hubert_base_librispeech


# Prepared Data Directory

data_dir=librispeech
# -- data_dir
#    -- test.tsv
#    -- test.ltr
#    -- dict.ltr.txt


exp_dir=exp     # Target experiments directory (where you have your pre-trained model with checkpoint_best.pt)
ratios="[1, 2]" # Default label rate ratios

_opts=

# If use slurm, uncomment this line and modify the job submission at
# _opts="${_opts} hydra/launcher=submitit_slurm +hydra.launcher.partition=${your_slurm_partition} +run=submitit_reg"

# If want to set additional experiment tag, uncomment this line
# _opts="${_opts} hydra.sweep.subdir=${your_experiment_tag}"

# If use un-normalized audio, uncomment this line
# _opts="${_opts} task.normalize=false"



PYTHONPATH=${FAIRSEQ}
python examples/speech_recognition/new/infer.py \
  --config-dir ${config_dir} \
  --config-name infer_multires \
   ${_opts} \
  task.data=${data_dir}  \
  task.label_rate_ratios='${ratios}' \
  common_eval.results_path=${exp_dir} \
  common_eval.path=${exp_dir}/checkpoint_best.pt \
  dataset.max_tokens=2000000 \
  dataset.gen_subset=test \
  dataset.skip_invalid_size_inputs_valid_test=true

