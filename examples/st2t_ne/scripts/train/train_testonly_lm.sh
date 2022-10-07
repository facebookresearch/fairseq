#!/bin/bash

l=$1
save_dir=/checkpoint/mgaido/2022/ne/lm/testonly/$l

mkdir -p $save_dir

# 4 GPU
fairseq-train /large_experiments/ust/mgaido/2022/data/nelm/ \
  --train-subset test.$l-None.$l --valid-subset test.$l-None.$l \
  --task language_modeling \
  --save-dir $save_dir \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 1024 --update-freq 1 \
  --max-update 20000 --keep-last-epochs 10 --patience 5 > $save_dir/train_short.log 2> $save_dir/train_short.err

python /private/home/mgaido/fairseq/scripts/average_checkpoints.py --input /checkpoint/mgaido/2022/ne/lm/testonly/$l/ --num-epoch-checkpoints 10 --output /checkpoint/mgaido/2022/ne/lm/testonly/$l/avg10_best.pt

