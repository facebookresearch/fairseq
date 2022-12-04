#!/usr/bin/env zsh

# if [ -z "$USER_DIR_OVERRIDE" ]; then
#   echo "please set USER_DIR_OVERRIDE"
#   exit 1
# fi

job_id=$1
task_id=$2
dir="$3"
cp="$dir/checkpoints/checkpoint_last.pt"

echo "job_id: $job_id, task_id: $task_id, dir: $dir"

PYTHONPATH=.:./examples nohup python fairseq_cli/hydra_train.py -m \
--config-dir examples/data2vec/config/vision/finetuning --config-name mae_imagenet hydra/launcher=submitit_slurm \
+run_config=slurm_2_aws_arbabu distributed_training.distributed_port=$[${RANDOM}%30000+30000] +task.local_cache_path=/scratch/cache_arbabu/imagenet \
common.user_dir=/data/home/arbabu/d2v_working_PUSH_blockM/fairseq-py/examples/data2vec common.fp16=true \
model.model_path=$cp distributed_training.find_unused_parameters=true \
+hydra.launcher.additional_parameters.dependency="afterok:$job_id" \
+task.rebuild_batches=true hydra.sweep.dir="$dir/finetune/last" hydra.launcher.name=vision_ft_last_${PREFIX} &