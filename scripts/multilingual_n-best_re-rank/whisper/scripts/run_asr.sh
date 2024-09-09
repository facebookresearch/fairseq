models=("tiny" "large-v2" "base" "small" "medium")
files=("fl" "babel")

for model in "${models[@]}"; do
    for file in "${files[@]}"; do
        srun --gpus-per-node=1 --cpus-per-task=4 --mem=10G --time=72:00:00 python scripts/infer_asr.py --wavs dump/$file/wav.txt --lids dump/$file/test.lang --dst exp_asr/$model/$file/oracle --model $model & \
        srun --gpus-per-node=1 --cpus-per-task=4 --mem=10G --time=72:00:00 python scripts/infer_asr.py --wavs dump/$file/wav.txt --lids exp_lid/$model/$file/lid.txt --dst exp_asr/$model/$file/lid-dep --model $model
    done
done