#!/bin/bash

# data2vec speed benchmark (only train 100K for speed estimate)
d2v_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/text/pretraining --config-name base_aws"
d2v_cmd="$d2v_cmd hydra/launcher=submitit_slurm +run_config=slurm_2_aws"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# PREFIX=d2v2/base_d2v1_speed/v1_default PYTHONPATH=$(pwd) $d2v_cmd $port &
# PREFIX=d2v2/base_d2v1_speed/v1_default_complete PYTHONPATH=$(pwd) $d2v_cmd $port task.sample_break_mode=complete &
# PREFIX=d2v2/base_d2v1_speed/v1_default_complete_6wrk PYTHONPATH=$(pwd) $d2v_cmd $port task.sample_break_mode=complete dataset.num_workers=6 &


# data2vec2 speed benchmark
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi --config-name base_text_only_task"
d2v2_cmd="$d2v2_cmd hydra/launcher=depedency_submitit_slurm +run_config=slurm_2_aws +next_script=examples/data2vec/scripts/multi/finetune_all_fair_aws_local_lr.sh"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"
# 
# PREFIX=d2v2/base_multi_speed/v1_default PYTHONPATH=$(pwd) $d2v2_cmd $port &
# PREFIX=d2v2/base_multi_speed/v1_breakmode PYTHONPATH=$(pwd) $d2v2_cmd $port task.sample_break_mode=complete &
# 
# PREFIX=d2v2/base_multi_speed/v1_update PYTHONPATH=$(pwd) $d2v2_cmd $port optimization.max_update=200000 &
# PREFIX=d2v2/base_multi_speed/v1_update PYTHONPATH=$(pwd) $d2v2_cmd $port optimization.max_update=400000 &
# PREFIX=d2v2/base_multi_speed/v1_update PYTHONPATH=$(pwd) $d2v2_cmd $port optimization.max_update=600000 &
# PREFIX=d2v2/base_multi_speed/v1_update PYTHONPATH=$(pwd) $d2v2_cmd $port optimization.max_update=800000 &
# 
# PREFIX=d2v2/base_multi_speed/v1_clone PYTHONPATH=$(pwd) $d2v2_cmd $port model.clone_batch=4 &
# PREFIX=d2v2/base_multi_speed/v1_clone PYTHONPATH=$(pwd) $d2v2_cmd $port model.clone_batch=2 &
# 
# PREFIX=d2v2/base_multi_speed/v1_clone_bs PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 &
# PREFIX=d2v2/base_multi_speed/v1_clone_bs PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=4 &
# PREFIX=d2v2/base_multi_speed/v1_clone_bs PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=2 &
# 
# PREFIX=d2v2/base_multi_speed/v1_maskp PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_prob=0.64 &
# PREFIX=d2v2/base_multi_speed/v1_maskp PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_prob=0.8 &

# PREFIX=d2v2/base_multi_speed/v2_clone_bs PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=8 model.clone_batch=2 &
# PREFIX=d2v2/base_multi_speed/v2_clone_bs PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=16 model.clone_batch=1 &
# PREFIX=d2v2/base_multi_speed/v2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 'optimizer.groups.default.lr_float=0.0001' &
# PREFIX=d2v2/base_multi_speed/v2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 'optimizer.groups.default.lr_float=0.0003' &
# PREFIX=d2v2/base_multi_speed/v2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 'optimizer.groups.default.lr_float=0.0004' &
# PREFIX=d2v2/base_multi_speed/v2_ema_start PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_decay=0.995 &
# PREFIX=d2v2/base_multi_speed/v2_ema_start PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_decay=0.999 &
# PREFIX=d2v2/base_multi_speed/v2_ema_start PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_decay=0.9995 &
# PREFIX=d2v2/base_multi_speed/v2_ema_end PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 &
# PREFIX=d2v2/base_multi_speed/v2_ema_step PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_anneal_end_step=200000 &
# PREFIX=d2v2/base_multi_speed/v2_ema_step PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_anneal_end_step=50000 &


# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 optimization.max_update=200000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 optimization.max_update=400000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 optimization.max_update=600000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 optimization.max_update=800000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 optimization.max_update=1200000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 optimization.max_update=1400000 &
# PREFIX=d2v2/base_multi_speed/v3_maskp PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.42 &
# PREFIX=d2v2/base_multi_speed/v3_maskp PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.54 &

PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.36 optimization.max_update=200000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.36 optimization.max_update=400000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.36 optimization.max_update=600000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.36 optimization.max_update=800000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.36 optimization.max_update=1200000 &
# PREFIX=d2v2/base_multi_speed/v3_upd PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.36 optimization.max_update=1400000 &
# PREFIX=d2v2/base_multi_speed/v3_maskp PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=2 model.clone_batch=8 model.ema_end_decay=1 model.modalities.text.mask_prob=0.36 &
# 
