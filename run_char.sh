#!/bin/bash

set -eu

# ==================== d2v2 char sweep 1
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/text/pretraining --config-name base_dual_mae_conv_char_aws"
d2v2_cmd="$d2v2_cmd hydra/launcher=depedency_submitit_slurm +next_script=examples/data2vec/scripts/text/finetune_all_char_fair_aws_local_lr.sh +run_config=slurm_2_aws"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# # baseline
# PREFIX=d2v2/char_base_16g_400K_swp1_baseline PYTHONPATH=$(pwd) $d2v2_cmd $port &

# # bs
# PREFIX=d2v2/char_base_16g_400K_swp2_bs PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 &

# # mask
# PREFIX=d2v2/char_base_16g_400K_swp2_mask PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 task.mask_prob=0.24 task.mask_multiple_length=4,8,12 &
# PREFIX=d2v2/char_base_16g_400K_swp2_mask PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 task.mask_prob=0.48 task.mask_multiple_length=2,4,8,12 &

# ==================== d2v2 char sweep 1 (4 node)
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/text/pretraining --config-name base_dual_mae_conv_char_aws"
d2v2_cmd="$d2v2_cmd hydra/launcher=depedency_submitit_slurm +next_script=examples/data2vec/scripts/text/finetune_all_char_fair_aws_local_lr.sh +run_config=slurm_4_aws"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# # bs
# PREFIX=d2v2/char_base_16g_400K_swp2_bs_4n PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 &
