#!/bin/bash

set -eu


# ==================== roberta
cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/roberta/config/pretraining --config-name base"
cmd="$cmd hydra/launcher=depedency_submitit_slurm +run_config=slurm_2_aws +next_script=examples/data2vec/scripts/text/finetune_all_fair_aws_local_lr.sh"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# PREFIX=d2v2/base_roberta/bpe/400k PYTHONPATH=$(pwd) $cmd $port optimization.max_update=400000 'optimization.lr=[3e-4]' &
# PREFIX=d2v2/base_roberta/bpe/1M PYTHONPATH=$(pwd) $cmd $port optimization.max_update=1000000 dataset.batch_size=16,32 'optimization.lr=[3e-4]' &
PREFIX=d2v2/base_roberta/bpe/1M PYTHONPATH=$(pwd) $cmd $port optimization.max_update=1000000 dataset.batch_size=32 'optimization.lr=[3e-4]' &


