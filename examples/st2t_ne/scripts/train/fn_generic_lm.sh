#!/bin/bash

l=$1
save_dir=/checkpoint/mgaido/2022/ne/generic_lm/$l/fn

mkdir -p $save_dir

if [ $l == "en" ]; then
  tgtl="es"
else
  tgtl=$l
fi

# 4 GPU
fairseq-train /large_experiments/ust/mgaido/2022/data/mt/fn/ \
  --train-subset train.en-$tgtl.$l --valid-subset valid.en-$tgtl.$l \
  --task language_modeling \
  --save-dir $save_dir \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none --restore-file /checkpoint/mgaido/2022/ne/generic_lm/$l/avg10_best.pt \
  --max-tokens 1024 --update-freq 4 --dataset-impl mmap \
  --max-update 200000 --keep-last-epochs 10 --patience 5 > $save_dir/train_short.log 2> $save_dir/train_short.err

python /private/home/mgaido/fairseq/scripts/average_checkpoints.py --input /checkpoint/mgaido/2022/ne/generic_lm/$l/fn --num-epoch-checkpoints 5 --output /checkpoint/mgaido/2022/ne/generic_lm/$l/fn/avg5_best.pt

rm /checkpoint/mgaido/2022/ne/generic_lm/$l/fn/checkpoint*
