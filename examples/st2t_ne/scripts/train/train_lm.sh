#!/bin/bash

l=$1
save_dir=/checkpoint/mgaido/2022/ne/lm/$l

mkdir -p $save_dir

# LM task does not support multiple train splits.... Hacking it
ln -s /large_experiments/ust/mgaido/2022/data/nelm/test.$l-None.$l.bin /large_experiments/ust/mgaido/2022/data/nelm/train.$l-None.${l}1.bin
ln -s /large_experiments/ust/mgaido/2022/data/nelm/test.$l-None.$l.idx /large_experiments/ust/mgaido/2022/data/nelm/train.$l-None.${l}1.idx
# 4 GPU
fairseq-train /large_experiments/ust/mgaido/2022/data/nelm/ \
  --train-subset train.$l-None.$l --valid-subset valid.$l-None.$l \
  --task language_modeling \
  --save-dir $save_dir \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 1024 --update-freq 4 \
  --max-update 20000 --keep-last-epochs 10 --patience 5 > $save_dir/train_short.log 2> $save_dir/train_short.err

python /private/home/mgaido/fairseq/scripts/average_checkpoints.py --input /checkpoint/mgaido/2022/ne/lm/$l/ --num-epoch-checkpoints 10 --output /checkpoint/mgaido/2022/ne/lm/$l/avg10_best.pt

rm /large_experiments/ust/mgaido/2022/data/nelm/train.$l-None.${l}1.bin
rm /large_experiments/ust/mgaido/2022/data/nelm/train.$l-None.${l}1.idx

rm /checkpoint/mgaido/2022/ne/lm/$l/checkpoint*
