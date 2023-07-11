#!/bin/bash

if [ $# -ne 5 ]; then
    echo "usage: $0 [dataset=wmt14/full] [langpair=en-de] [databin] [bpecode] [model]"
    exit
fi


DATASET=$1
LANGPAIR=$2
DATABIN=$3
BPECODE=$4
MODEL=$5

SRCLANG=$(echo $LANGPAIR | cut -d '-' -f 1)
TGTLANG=$(echo $LANGPAIR | cut -d '-' -f 2)


BPEROOT=examples/backtranslation/subword-nmt/subword_nmt
if [ ! -e $BPEROOT ]; then
    BPEROOT=subword-nmt/subword_nmt
    if [ ! -e $BPEROOT ]; then
        echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
        git clone https://github.com/rsennrich/subword-nmt.git
    fi
fi


TMP_REF=$(mktemp)

sacrebleu -t $DATASET -l $LANGPAIR --echo ref -q \
| sacremoses normalize -l $TGTLANG -q \
| sacremoses tokenize -a -l $TGTLANG -q \
> $TMP_REF

sacrebleu -t $DATASET -l $LANGPAIR --echo src -q \
| sacremoses normalize -l $SRCLANG -q \
| sacremoses tokenize -a -l $SRCLANG -q \
| python $BPEROOT/apply_bpe.py -c $BPECODE \
| fairseq-interactive $DATABIN --path $MODEL \
    -s $SRCLANG -t $TGTLANG \
    --beam 5 --remove-bpe --buffer-size 1024 --max-tokens 8000 \
| grep ^H- | cut -f 3- \
| fairseq-score --ref $TMP_REF

rm -f $TMP_REF
