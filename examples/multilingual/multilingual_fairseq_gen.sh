#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

lang_pairs="en-fr,en-cs,fr-en,cs-en"
path_2_data=$1 # <path to data>
lang_list=$2 # <path to a file which contains list of languages separted by new lines>
model=$3  # <path to a trained model>
source_lang=cs
target_lang=en

fairseq-generate "$path_2_data" \
  --path "$model" \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang "$source_lang" \
  --target-lang "$target_lang" \
  --sacrebleu --remove-bpe 'sentencepiece'\
  --batch-size 32 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs"
