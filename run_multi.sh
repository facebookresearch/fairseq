#!/bin/bash

set -eu

# ==================== sweep 1
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi --config-name base_text_only_task"
d2v2_cmd="$d2v2_cmd hydra/launcher=submitit_slurm +run_config=slurm_2_aws"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# baseline
# PREFIX=d2v2/multi_text_only_base_16g_400K_swp1_baseline_prenet0_0926 PYTHONPATH=$(pwd) $d2v2_cmd $port &
# PREFIX=d2v2/multi_text_only_base_16g_400K_swp1_baseline_prenet0_0926_lr PYTHONPATH=$(pwd) $d2v2_cmd $port 'optimization.lr=[3e-3],[1e-3],[5e-4]'&

# ==================== sweep 2
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi --config-name base_text_only_task_pgrp"
d2v2_cmd="$d2v2_cmd hydra/launcher=depedency_submitit_slurm +run_config=slurm_2_aws +next_script=examples/data2vec/scripts/multi/finetune_all_fair_aws_local_lr.sh"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# # baseline
# PREFIX=d2v2/base_multi/bpe/repr_text PYTHONPATH=$(pwd) $d2v2_cmd $port &
# PREFIX=d2v2/base_multi/bpe/repr_text_lscale PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_scale=0.1,0.01,0.001 &

# # swp1
# PREFIX=d2v2/base_multi/bpe/swp1_inv-mask PYTHONPATH=$(pwd) $d2v2_cmd $port \
#   +model.modalities.text.inverse_mask=True model.modalities.text.mask_prob=0.24,0.36,0.48,0.64 \
#   model.modalities.text.mask_length=2,4 &
# 
# PREFIX=d2v2/base_multi/bpe/swp1_fix-alibi PYTHONPATH=$(pwd) $d2v2_cmd $port \
#   +model.modalities.text.use_alibi_encoder=True +model.modalities.text.alibi_scale=0.5,1.0,1.5,2.0 \
#   +model.modalities.text.alibi_max_pos=512 &

# PREFIX=d2v2/base_multi/bpe/swp1_learn-alibi PYTHONPATH=$(pwd) $d2v2_cmd $port \
#   +model.modalities.text.use_alibi_encoder=True +model.modalities.text.alibi_max_pos=512 \
#   +model.modalities.text.learned_alibi=True +model.modalities.text.learned_alibi_scale=True,False \
#   +model.modalities.text.learned_alibi_scale_per_head=True,False \
#   +model.modalities.text.learned_alibi_scale_per_layer=True,False &

# PREFIX=d2v2/base_multi/bpe/swp1_tgtnorm PYTHONPATH=$(pwd) $d2v2_cmd $port model.layer_norm_target_layer=True model.instance_norm_target_layer=False model.batch_norm_target_layer=False model.instance_norm_targets=False model.layer_norm_targets=False,True &
# PREFIX=d2v2/base_multi/bpe/swp1_tgtnorm PYTHONPATH=$(pwd) $d2v2_cmd $port model.layer_norm_target_layer=False model.instance_norm_target_layer=True model.batch_norm_target_layer=False model.instance_norm_targets=False model.layer_norm_targets=False,True &
# PREFIX=d2v2/base_multi/bpe/swp1_tgtnorm PYTHONPATH=$(pwd) $d2v2_cmd $port model.layer_norm_target_layer=False model.instance_norm_target_layer=False model.batch_norm_target_layer=True model.instance_norm_targets=False model.layer_norm_targets=False,True &
# PREFIX=d2v2/base_multi/bpe/swp1_tgtnorm PYTHONPATH=$(pwd) $d2v2_cmd $port model.layer_norm_target_layer=False model.instance_norm_target_layer=True model.batch_norm_target_layer=False model.instance_norm_targets=True model.layer_norm_targets=False &
# PREFIX=d2v2/base_multi/bpe/swp1_tgtnorm PYTHONPATH=$(pwd) $d2v2_cmd $port model.layer_norm_target_layer=False model.instance_norm_target_layer=False model.batch_norm_target_layer=True model.instance_norm_targets=True model.layer_norm_targets=False &

# ==================== sweep 1 (1M)
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi --config-name base_text_only_task_pgrp_1M"
d2v2_cmd="$d2v2_cmd hydra/launcher=depedency_submitit_slurm +run_config=slurm_2_aws +next_script=examples/data2vec/scripts/multi/finetune_all_fair_aws_local_lr.sh"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# PREFIX=d2v2/base_multi/bpe/1M_swp1_baseline PYTHONPATH=$(pwd) $d2v2_cmd $port &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_maskp PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_prob=0.3,0.4,0.6,0.7 &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=4e-4 optimizer.groups.decoder.lr_float=4e-3 &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=3e-4 optimizer.groups.decoder.lr_float=3e-3 &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=2e-4 optimizer.groups.decoder.lr_float=2e-3 &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_ema-anl PYTHONPATH=$(pwd) $d2v2_cmd $port model.ema_anneal_end_step=50000,100000,150000 &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_dec PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.decoder.input_dropout=0,0.05,0.15 &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_dec PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.decoder.decoder_dim=256,512,1024 &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_dec PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.decoder.decoder_residual=true &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_dec PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.decoder.projection_layers=1,3 &
# PREFIX=d2v2/base_multi/bpe/1M_swp1_dec PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.decoder.decoder_kernel=3,5,7,11,15 &


# ==================== sweep 2 (1M)
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=1 model.modalities.text.mask_prob=0.48 dataset.batch_size=2 model.clone_batch=16,32 &
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=1 model.modalities.text.mask_prob=0.48 dataset.batch_size=4 model.clone_batch=8,16 &
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=1 model.modalities.text.mask_prob=0.48 dataset.batch_size=8 model.clone_batch=4 &
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=1 model.modalities.text.mask_prob=0.48 dataset.batch_size=8 model.clone_batch=8 &
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=1 model.modalities.text.mask_prob=0.48 dataset.batch_size=16 model.clone_batch=8 &
#                                                                                                                                           
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=2 model.modalities.text.mask_prob=0.6 dataset.batch_size=2 model.clone_batch=16,32 &
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=2 model.modalities.text.mask_prob=0.6 dataset.batch_size=4 model.clone_batch=8,16 &
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=2 model.modalities.text.mask_prob=0.6 dataset.batch_size=8 model.clone_batch=4 &
#                                                                                                                                           
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=2 model.modalities.text.mask_prob=0.36 +model.modalities.text.inverse_mask=True dataset.batch_size=2 model.clone_batch=16,32 &
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=2 model.modalities.text.mask_prob=0.36 +model.modalities.text.inverse_mask=True dataset.batch_size=4 model.clone_batch=8,16 &
#                                                                                                                                           
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=4 model.modalities.text.mask_prob=0.24 +model.modalities.text.inverse_mask=True dataset.batch_size=2 model.clone_batch=16,32 &
# PREFIX=d2v2/base_multi/bpe/1M_swp2_bs-and-mask PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_length=4 model.modalities.text.mask_prob=0.24 +model.modalities.text.inverse_mask=True dataset.batch_size=4 model.clone_batch=8,16 &

# ==================== large 
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi --config-name large_text_only_task_pgrp_1M"
d2v2_cmd="$d2v2_cmd hydra/launcher=depedency_submitit_slurm +run_config=slurm_4_aws +next_script=examples/data2vec/scripts/multi/finetune_all_fair_aws_local_lr.sh"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# # 400K
# PREFIX=d2v2/large_multi/bpe/400K_swp1_baseline PYTHONPATH=$(pwd) $d2v2_cmd $port optimization.max_update=400000 model.ema_anneal_end_step=10000 &

# # 1M
# PREFIX=d2v2/large_multi/bpe/1M_swp1_baseline PYTHONPATH=$(pwd) $d2v2_cmd $port &

# PREFIX=d2v2/large_multi/bpe/1M_swp2_mode PYTHONPATH=$(pwd) $d2v2_cmd $port task.sample_break_mode=complete &
# 
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=4e-4 optimizer.groups.decoder.lr_float=4e-3 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=3e-4 optimizer.groups.decoder.lr_float=3e-3 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=2e-4 optimizer.groups.decoder.lr_float=2e-3 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=1e-4 optimizer.groups.decoder.lr_float=1e-4 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=1e-3 optimizer.groups.decoder.lr_float=1e-3 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=2e-4 optimizer.groups.decoder.lr_float=2e-4 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=3e-4 optimizer.groups.decoder.lr_float=3e-4 &
# 
# PREFIX=d2v2/large_multi/bpe/1M_swp2_warm PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_scheduler.warmup_updates=8000 optimizer.groups.decoder.lr_scheduler.warmup_updates=8000 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_warm PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_scheduler.warmup_updates=16000 optimizer.groups.decoder.lr_scheduler.warmup_updates=16000 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_warm PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_scheduler.warmup_updates=32000 optimizer.groups.decoder.lr_scheduler.warmup_updates=32000 &
# 
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_beta=0,0.1,1,10 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_scale=5,1,2,0.5,0 &
# 
# PREFIX=d2v2/large_multi/bpe/1M_swp2_bs PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=4 model.clone_batch=4 &
# 
# PREFIX=d2v2/large_multi/bpe/1M_swp2_ema PYTHONPATH=$(pwd) $d2v2_cmd $port model.ema_decay=0.9999 model.ema_end_decay=0.99995,0.999999 model.ema_anneal_end_step=50000,100000,200000 &
# PREFIX=d2v2/large_multi/bpe/1M_swp2_ema PYTHONPATH=$(pwd) $d2v2_cmd $port model.ema_decay=0.9995 model.ema_end_decay=0.99995,0.999999 model.ema_anneal_end_step=50000,100000,200000 &

# ================= swp3 lr=1e-4
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi --config-name large_text_only_task_pgrp_1M"
d2v2_cmd="$d2v2_cmd hydra/launcher=depedency_submitit_slurm +run_config=slurm_4_aws +next_script=examples/data2vec/scripts/multi/finetune_all_fair_aws_local_lr.sh"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# TODO: rerun below. canceled due to speed issue
# PREFIX=d2v2/large_multi/bpe/1M_swp3_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_beta=0 &
# PREFIX=d2v2/large_multi/bpe/1M_swp3_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_beta=0.1 &
# PREFIX=d2v2/large_multi/bpe/1M_swp3_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_beta=1 &

# PREFIX=d2v2/large_multi/bpe/1M_swp3_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_scale=0.1 &
# PREFIX=d2v2/large_multi/bpe/1M_swp3_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_scale=0.01 &
# PREFIX=d2v2/large_multi/bpe/1M_swp3_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_scale=1 &
# PREFIX=d2v2/large_multi/bpe/1M_swp3_lossbeta PYTHONPATH=$(pwd) $d2v2_cmd $port model.loss_scale=null &

# PREFIX=d2v2/large_multi/bpe/1M_swp3_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=2e-4 optimizer.groups.decoder.lr_float=2e-4 &

# ================= swp4
d2v2_cmd="nohup python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi --config-name large_text_only_task"
d2v2_cmd="$d2v2_cmd hydra/launcher=depedency_submitit_slurm +run_config=slurm_4_aws +next_script=examples/data2vec/scripts/multi/finetune_all_fair_aws_local_lr.sh"
port="distributed_training.distributed_port=$[${RANDOM}%30000+30000]"

# PREFIX=d2v2/large_multi/bpe/600K_swp4_baseline PYTHONPATH=$(pwd) $d2v2_cmd $port &
# PREFIX=d2v2/large_multi/bpe/600K_swp4_bs PYTHONPATH=$(pwd) $d2v2_cmd $port dataset.batch_size=4 &
# PREFIX=d2v2/large_multi/bpe/600K_swp4_lr PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_float=0.00015,0.00012,0.00008 &
# PREFIX=d2v2/large_multi/bpe/600K_swp4_lrwarm PYTHONPATH=$(pwd) $d2v2_cmd $port optimizer.groups.default.lr_scheduler.warmup_updates=8000,12000 &
# PREFIX=d2v2/large_multi/bpe/600K_swp4_clone PYTHONPATH=$(pwd) $d2v2_cmd $port model.clone_batch=4,6,10 &
# PREFIX=d2v2/large_multi/bpe/600K_swp4_ema_start PYTHONPATH=$(pwd) $d2v2_cmd $port model.ema_decay=0.9997,0.9998 &
# PREFIX=d2v2/large_multi/bpe/600K_swp4_ema_end PYTHONPATH=$(pwd) $d2v2_cmd $port model.ema_end_decay=1 &
# PREFIX=d2v2/large_multi/bpe/600K_swp4_topk PYTHONPATH=$(pwd) $d2v2_cmd $port model.average_top_k_layers=22,20,18 &
# PREFIX=d2v2/large_multi/bpe/600K_swp4_maskp PYTHONPATH=$(pwd) $d2v2_cmd $port model.modalities.text.mask_prob=0.48,0.42,0.54 &


# =============================
# # test pretrain
# PYTHONPATH=.:./examples PREFIX=d2v/libri/multi_final_200k python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi --config-name base_text_only_task +run_config=local optimization.max_update=20
# PYTHONPATH=.:./examples PREFIX=d2v/libri/multi_final_200k python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/text/pretraining --config-name base_dual_mae_conv_v3_aws +run_config=local optimization.max_update=20

# # test finetune
# PYTHONPATH=. PREFIX="dummy" SUFFIX="" python fairseq_cli/hydra_train.py -m --config-dir examples/roberta/config/finetuning --config-name sst_2 +run_config=local task.data="/fsx-wav2vec/abaevski/data/nlp/GLUE/SST-2-bin" checkpoint.restore_file="/fsx-wav2vec/wnhsu/d2v2/full_swp3_lr/base_dual_mae_conv_v3_aws_8/optimizer.groups.decoder.lr_float_0.0009/optimizer.groups.encoder.lr_float_9e-05/checkpoints/checkpoint_last.pt"
# PYTHONPATH=. PREFIX="dummy" SUFFIX='' python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/multi/text_finetuning --config-name cola +run_config=local task.data=/fsx-wav2vec/abaevski/data/nlp/GLUE/CoLA-bin dataset.num_workers=1 model.model_path=/fsx-wav2vec/wnhsu/d2v2/base_multi/bpe/repr_text/base_text_only_task_pgrp_8/checkpoints/checkpoint_last.pt hydra.sweep.dir=$PWD/tmp_dbg +model=text_wrap

# # test validate
# PYTHONPATH=.:./examples PREFIX=d2v/libri/multi_final_200k python fairseq_cli/hydra_validate.py --config-dir examples/data2vec/config/multi --config-name base_text_only_task_pgrp +run_config=local common_eval.path=/fsx-wav2vec/wnhsu/d2v2/base_multi/bpe/1M_swp1_baseline/base_text_only_task_pgrp_1M_8/finetune_lr/cola/1e-5/0/checkpoints/checkpoint_best.pt task.data=/fsx-wav2vec/abaevski/data/nlp/GLUE/CoLA-bin dataset.batch_size=64
