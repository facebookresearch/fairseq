#!/usr/bin/env bash

# usage: bash binarize_manifest <dest_dir> <train_split> <valid_split>

DEST_DIR=$1
TRAIN_SPLIT=$2
VALID_SPLIT=$3
FAIRSEQ_ROOT=$4

mkdir -p $DEST_DIR

# split file path and lengths into separate files
cut -f1 $TRAIN_SPLIT.tsv > $DEST_DIR/train_fnames.txt
cut -f1 $VALID_SPLIT.tsv > $DEST_DIR/valid_fnames.txt
cut -f2 $TRAIN_SPLIT.tsv > $DEST_DIR/train.lengths
cut -f2 $VALID_SPLIT.tsv > $DEST_DIR/valid.lengths

# copy root directory
head -1 $TRAIN_SPLIT.tsv > $DEST_DIR/train.root
head -1 $VALID_SPLIT.tsv > $DEST_DIR/valid.root

# remove root directory
sed -i '1d' $DEST_DIR/train_fnames.txt
sed -i '1d' $DEST_DIR/valid_fnames.txt
sed -i '1d' $DEST_DIR/train.lengths
sed -i '1d' $DEST_DIR/valid.lengths

# insert spaces between characters
sed -i -e 's/\(.\)/\1 /g' $DEST_DIR/train_fnames.txt
sed -i -e 's/\(.\)/\1 /g' $DEST_DIR/valid_fnames.txt

# run preprocessor
PYTHONPATH=$FAIRSEQ_ROOT python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $DEST_DIR/train_fnames.txt --validpref $DEST_DIR/valid_fnames.txt --workers 60 --only-source --destdir $DEST_DIR
