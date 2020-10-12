#!/usr/bin/env bash

# shellcheck disable=SC2164
python -m pip install --editable .
python python_path.py
echo $PYTHONPATH
pip install fastBPE sacremoses subword_nmt

cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

TEXT=examples/translation/iwslt14.tokenized.de-en && \
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --fp16 --max-epoch 1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt --fp16 \
    --batch-size 128 --beam 5 --remove-bpe True --quiet
