#!/bin/bash

set -eu

# ==================== 

d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/text/pretraining --config-name large_dual_mae_conv_aws hydra/launcher=depedency_submitit_slurm +next_script=examples/data2vec/scripts/text/finetune_all_large_fair_aws_local_lr.sh +run_config=slurm_4_aws"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# # baseline
# PREFIX=d2v2/large_32g_400K_swp1_baseline PYTHONPATH=$(pwd) $d2v2_cmd $port &

# # LR
# PREFIX=d2v2/large_32g_400K_swp1_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.encoder.lr_float=1e-4 optimizer.groups.decoder.lr_float=9e-4,3e-3 &
# PREFIX=d2v2/large_32g_400K_swp1_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.encoder.lr_float=9e-5 optimizer.groups.decoder.lr_float=9e-4,1e-3,3e-3 &
# PREFIX=d2v2/large_32g_400K_swp1_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.encoder.lr_float=3e-4 optimizer.groups.decoder.lr_float=9e-4,1e-3,3e-3 &

# # Top-K
# PREFIX=d2v2/large_32g_400K_swp1_topk PYTHONPATH=$(pwd) $d2v2_cmd $port  model.average_top_k_layers=24,16,12 &
# PREFIX=d2v2/large_32g_400K_swp1_topk PYTHONPATH=$(pwd) $d2v2_cmd $port  model.average_top_k_layers=8,4 &


# ==================== 

d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/text/pretraining --config-name large_dual_mae_conv_aws hydra/launcher=depedency_submitit_slurm +next_script=examples/data2vec/scripts/text/finetune_all_large_fair_aws_local_lr.sh +run_config=slurm_8_aws"
# PREFIX=d2v2/large_64g_400K_swp1_baseline PYTHONPATH=$(pwd) $d2v2_cmd $port &
# PREFIX=d2v2/large_64g_400K_swp1_update PYTHONPATH=$(pwd) $d2v2_cmd $port optimization.max_update=600000 optimizer.groups.encoder.lr_scheduler.warmup_updates=6000 optimizer.groups.decoder.lr_scheduler.warmup_updates=6000 &
# PREFIX=d2v2/large_64g_400K_swp1_update PYTHONPATH=$(pwd) $d2v2_cmd $port optimization.max_update=800000 optimizer.groups.encoder.lr_scheduler.warmup_updates=8000 optimizer.groups.decoder.lr_scheduler.warmup_updates=8000 &
