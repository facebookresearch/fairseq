#!/bin/bash

save_dir=/checkpoint/mgaido/2022/ne/mt/base_multilang
mkdir -p $save_dir
# 4 GPU
fairseq-train /large_experiments/ust/mgaido/2022/data/mt/ \
    --max-epoch 100 --patience 20 \
    --task multilingual_translation --lang-pairs en-es,en-fr,en-it \
    --arch multilingual_transformer_iwslt_de_en \
    --ddp-backend no_c10d --num-workers 2 \
    --share-encoders --share-encoder-embeddings --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 16000 \
    --update-freq 1 --dataset-impl mmap \
    --save-dir $save_dir >> $save_dir/train.log 2> $save_dir/train.err

