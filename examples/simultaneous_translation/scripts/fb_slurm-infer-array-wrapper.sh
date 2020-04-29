model_dir=`realpath $2`
config=`realpath $1`
user_dir=`dirname $0`/..

cd $user_dir

num_checkpoints=`ls $model_dir/*.pt | grep -Po "checkpoint[0-9]+" | wc -l`
mkdir -p $model_dir/slurm-logs

sbatch \
    --gres=gpu:1 \
    --partition=learnfair \
    --time=00:10:00 \
    --cpus-per-task 4 \
    -e $model_dir/slurm-logs/infer-%A-%a.err \
    -o $model_dir/slurm-logs/infer-%A-%a.out \
    --array=1-$num_checkpoints \
    $user_dir/scripts/fb_fast-infer-array-wrapper.sh $config $model_dir
