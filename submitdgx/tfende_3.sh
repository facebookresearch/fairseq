#!/bin/bash

nvidia-smi
echo $PATH
which pip
which python3
echo "password" | sudo -S -k  apt-get update
echo "password" | sudo -S -k  apt-get install build-essential --assume-yes
echo "password" | sudo -S -k chmod 777 ../fairseq-1/
echo "password" | sudo -S env "PATH=$PATH" pip install --editable ./
echo "password" | sudo -S env "PATH=$PATH" pip install -r requirements.txt
#python3 main_mt.py -m transformer /-b 128  -d EN_DE -g 0 -layer 3 -s dgx #> energyLM-wiki2.out
echo "password" | sudo -S env "PATH=$PATH" python3 setup.py build_ext --inplace
echo "password" | sudo -S env "PATH=$PATH" python3 train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
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