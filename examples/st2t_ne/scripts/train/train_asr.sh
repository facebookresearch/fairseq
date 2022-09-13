#!/bin/bash

lang=$1
save_dir=/checkpoint/mgaido/2022/ne/asr/tagged/en-$lang
mkdir -p $save_dir
# 4 GPU
python /private/home/mgaido/fairseq/train.py /large_experiments/ust/mgaido/2022/data/s2t/en-es/ \
	--config-yaml config_asr.yaml --train-subset train_ep_asr_notags,train_mustc_asr_notags --valid-subset dev_ep_asr_notags \
	--num-workers 4 --max-tokens 20000 --max-update 100000 \
	--task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--arch s2t_wav_transformer_base --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 --clip-norm 10.0 --seed 1 --update-freq 4 \
	--load-pretrained-wav2vec-encoder /private/home/mgaido/models/wav2vec/wav2vec_small.pt \
	--save-dir $save_dir >> $save_dir/train.log 2> $save_dir/train.err

