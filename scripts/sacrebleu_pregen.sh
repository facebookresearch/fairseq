#!/bin/bash

if [ $# -ne 4 ]; then
    echo "usage: $0 TESTSET SRCLANG TGTLANG GEN"
    exit 1
fi

TESTSET=$1
SRCLANG=$2
TGTLANG=$3

GEN=$4

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=mosesdecoder/scripts
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl

grep ^H $GEN \
| sed 's/^H\-//' \
| sort -n -k 1 \
| cut -f 3 \
| perl $DETOKENIZER -l $TGTLANG \
| sed "s/ - /-/g" \
> $GEN.sorted.detok

sacrebleu --test-set $TESTSET --language-pair "${SRCLANG}-${TGTLANG}" < $GEN.sorted.detok
