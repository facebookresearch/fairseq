set=$1
type=$2
lids=$3
ref=$4
model=$5

# # data prep
# python examples/mms/asr/scripts/prep_lang_splits.py --wavs /private/home/yanb/whisper/dump/${set}/wav.txt --lids $lids --dst /private/home/yanb/MMS1_public/fairseq/examples/mms/asr/dump/${set}/${type}

# asr inference
srun --gpus-per-node=1 --cpus-per-task=4 --mem=10G --time=72:00:00 python examples/mms/asr/scripts/run_lang_splits.py --dump ~/MMS1_public/fairseq/examples/mms/asr/dump/${set}/${type}/ --model /checkpoint/yanb/MMS1_models/asr/$model/mms1b_l${model}.pt --dst ~/MMS1_public/fairseq/examples/mms/asr/exp/${model}/${set}/${type}/

# scoring
python examples/mms/asr/scripts/merge_split_hyps.py --exp examples/mms/asr/exp/${model}/${set}/${type}/ --dump examples/mms/asr/dump/${set}/${type}/
python examples/mms/asr/scripts/score.py --ref $ref --hyp examples/mms/asr/exp/${model}/${set}/${type}/hypo.units