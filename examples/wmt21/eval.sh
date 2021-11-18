#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
SRC=en
TGT=is
MODEL_NAME=wmt21.dense-24-wide.En-X

PATH_TO_FAIRSEQ_PY=.
TMP_DIR=generation_tmp
mkdir -p $TMP_DIR

REPLACE_UNICODE_PUNCT=$PATH_TO_FAIRSEQ_PY/examples/wmt21/scripts/replace-unicode-punctuation.perl
NORM_PUNCT=$PATH_TO_FAIRSEQ_PY/examples/wmt21/scripts/normalize-punctuation.perl
if [ ! -d "${TMP_DIR}/${MODEL_NAME}" ]; then
  wget https://dl.fbaipublicfiles.com/fairseq/models/${MODEL_NAME}.tar.gz -P $TMP_DIR/
  tar -xvf $TMP_DIR/${MODEL_NAME}.tar.gz -C $TMP_DIR
fi
MODEL_DIR=$TMP_DIR/${MODEL_NAME}
if [ ! -d "${TMP_DIR}/wmt21-news-systems" ]; then
  git clone https://github.com/wmt-conference/wmt21-news-systems $TMP_DIR/wmt21-news-systems
fi

DOMAIN_TAG="wmtdata newsdomain"
INPUT_FILE=$TMP_DIR/wmt21-news-systems/txt/sources/newstest2021.${SRC}-${TGT}.src.${SRC}
REF_FILE=$TMP_DIR/wmt21-news-systems/txt/references/newstest2021.${SRC}-${TGT}.ref.A.${TGT}

# Translate
cat ${INPUT_FILE} | sed "s/^/${DOMAIN_TAG} /" | $REPLACE_UNICODE_PUNCT | $NORM_PUNCT -l ${SRC} | python $PATH_TO_FAIRSEQ_PY/fairseq_cli/interactive.py  $MODEL_DIR \
  --path ${MODEL_DIR}/checkpoint.pt \
  --task translation_multi_simple_epoch \
  --langs "en,ha,is,ja,cs,ru,zh,de" \
  --lang-pairs $SRC-$TGT \
  --bpe "sentencepiece" \
  --sentencepiece-model ${MODEL_DIR}/sentencepiece.model \
  --buffer-size 1024 \
  --batch-size 10 -s $SRC -t $TGT \
  --decoder-langtok \
  --encoder-langtok src  \
  --beam 5 \
  --lenpen 1.0 \
  --fp16  > $TMP_DIR/${SRC}-${TGT}.gen_log

cat $TMP_DIR/$SRC-$TGT.gen_log | grep -P "^D-" | cut -f3 > $TMP_DIR/$SRC-$TGT.hyp

# Calculate BLEU score
sacrebleu -l $SRC-$TGT $REF_FILE < $TMP_DIR/$SRC-$TGT.hyp
