#!/bin/bash

TESTSET=$1
SRCLANG=$2
TGTLANG=$3

GEN=$4

if [ $# -ne 4 ]; then
    echo "usage: $0 TESTSET SRCLANG TGTLANG GEN"
    exit 1
fi

NORM_PUNC=/private/home/felixwu/data/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
DETOKENIZER=/private/home/felixwu/data/mosesdecoder/scripts/tokenizer/detokenizer.perl

grep ^H $GEN \
| sed 's/^H\-//' \
| sort -n -k 1 \
| cut -f 3 \
| perl $DETOKENIZER -l $TGTLANG \
| sed "s/ - /-/g" \
> $GEN.sorted.detok
#| perl $NORM_PUNC $TGTLANG \

sacrebleu --test-set $TESTSET --language-pair "${SRCLANG}-${TGTLANG}" < $GEN.sorted.detok
