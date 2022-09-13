#!/bin/bash

save_dir=$1
# 4 GPU
python /private/home/mgaido/fairseq/train.py /large_experiments/ust/mgaido/2022/data/s2t/en-es/ \
	--config-yaml config_asr.yaml --train-subset train_ep_asr,train_mustc_asr --valid-subset dev_ep_asr \
	--num-workers 4 --max-tokens 20000 --max-update 100000 \
	--task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--arch s2t_transformer_m --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 --clip-norm 10.0 --seed 1 --update-freq 4 \
	--save-dir $save_dir >> $save_dir/train.log 2> $save_dir/train.err

