#!/bin/bash

path_2_data=$1  # <path to data> which contains binarized data for each directions
lang_list=$2  # <path to a file which contains a list of languages separted by new lines>
lang_pairs=$3  #a list language pairs to train multilingual models, e.g. "en-fr,en-cs,fr-en,cs-en"
# pretrained can be an mBART pretrained model as well
pretrained_model=$4 #<path to a pretrained model>


fairseq-train "$path_2_data" \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --finetune-from-model "$pretrained_model" \
  --sampling-method "temperature" \
  --sampling-temperature "1.5" \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2
