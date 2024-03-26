#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz"
    "http://data.statmt.org/wmt18/translation-task/rapid2016.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training-parallel-nc-v13/news-commentary-v13.de-en"
    "rapid2016.de-en"
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=de
lang=en-de
prep=wmt18_en_de
tmp=$prep/tmp
orig=orig
dev=dev/newstest2012
codes=32000
bpe=bpe.32k

mkdir -p $orig $tmp $prep $bpe

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    url=${URLS[i]}
    file=$(basename $url)
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit 1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm  -rf $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -l $l -no-escape >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l -no-escape > $tmp/test.$l
    echo ""
done

# apply length filtering before BPE
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train 1 100

# use newstest2012 for valid
echo "pre-processing valid data..."
for l in $src $tgt; do
    rm  -rf $tmp/valid.$l
    cat $orig/$dev.$l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -l $l -no-escape >> $tmp/valid.$l
done

mkdir output
mv $tmp/{train,valid,test}.{$src,$tgt} output

#BPE
git clone https://github.com/glample/fastBPE.git
pushd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
popd
fastBPE/fast learnbpe $codes output/train.$src output/train.$tgt > $bpe/codes
for split in {train,valid,test}; do for lang in {en,de}; do fastBPE/fast applybpe $bpe/$split.$lang output/$split.$lang $bpe/codes; done; done
