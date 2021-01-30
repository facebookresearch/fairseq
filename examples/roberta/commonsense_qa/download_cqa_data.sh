#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

OUTDIR=data/CommonsenseQA

mkdir -p $OUTDIR

wget -O $OUTDIR/train.jsonl https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
wget -O $OUTDIR/valid.jsonl https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
wget -O $OUTDIR/test.jsonl https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl
wget -O $OUTDIR/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
