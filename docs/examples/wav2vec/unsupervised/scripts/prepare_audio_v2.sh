#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

source_dir=$1
tgt_dir=$2
model=$3

if [ -z "$4" ]
  then
    dim=64
  else
    dim=$4
fi

echo "using $dim clusters for auxilary target"

if [ -z "$5" ]
  then
    layer=14
  else
    layer=$5
fi

echo "extracting from layer $layer"

train_split=train
valid_split=valid
test_split=test

all_splits=($train_split)

if [[ -f "$source_dir/valid.tsv" ]]; then
    all_splits+=('valid')
fi

if [[ -f "$source_dir/test.tsv" ]]; then
    all_splits+=('test')
fi

echo "processing splits: $all_splits"

mkdir -p $tgt_dir

cp $source_dir/*.tsv $tgt_dir
cp $source_dir/*.wrd $tgt_dir
cp $source_dir/*.ltr $tgt_dir
cp $source_dir/*.phn $tgt_dir
cp $source_dir/dict* $tgt_dir

setopt shwordsplit

for split in $all_splits; do
  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py $source_dir --split $split \
  --save-dir $tgt_dir --checkpoint $model --layer $layer
done


mkdir -p $tgt_dir/mfcc

# Consider spliting corpus into chuncks for large corpus, see HuBERT preprocessing for more details
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py \
  $tgt_dir $train_split 1 0 $tgt_dir/mfcc
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py \
  $tgt_dir/mfcc $train_split $tgt_dir/mfcc/cls$dim 1 0 $tgt_dir/mfcc/cls${dim}_idx
cp $tgt_dir/mfcc/cls${dim}_idx/${train_split}_0_1.km $tgt_dir/$train_split.km
