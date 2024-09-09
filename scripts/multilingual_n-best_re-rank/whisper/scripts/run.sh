model=$1

files=("vl" "babel" "vp" "mms-lab" "mms-unlab")
for file in "${files[@]}"; do
    srun --gpus-per-node=1 --cpus-per-task=4 --mem=10G --time=72:00:00 python scripts/infer_lid.py --wavs dump/$file/wav.txt --dst exp_lid/$model/$file/predictions.txt --model $model
done