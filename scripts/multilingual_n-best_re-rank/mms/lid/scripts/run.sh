set=$1
manifest=$2
n=$3

srun --gpus-per-node=1 --cpus-per-task=4 --mem=10G --time=3-00:00:00 python infer.py /checkpoint/yanb/MMS1_models/$n --path /checkpoint/yanb/MMS1_models/$n/mms1b_l${n}.pt --task audio_classification --infer-manifest ${manifest}.tsv --output-path exp/mms1b_l${n}/${set}
python scripts/score_lid.py --expdir exp/mms1b_l${n}/${set}/ --ref ${manifest}.label