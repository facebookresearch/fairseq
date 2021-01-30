#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
source_lang=kk_KZ
target_lang=en_XX
MODEL=criss_checkpoints/criss.3rd.pt
SPM=criss_checkpoints/sentence.bpe.model
SPLIT=test
LANG_DICT=criss_checkpoints/lang_dict.txt
SPM_ENCODE=flores/scripts/spm_encode.py
SAVE_ENCODER=save_encoder.py
ENCODER_SAVE_ROOT=sentence_embeddings/$MODEL
DICT=criss_checkpoints/dict.txt
THRESHOLD=1.02
MIN_COUNT=500

DATA_DIR=data_tmp
SAVE_DIR=mining/${source_lang}_${target_lang}_mined
ENCODER_SAVE_DIR=${ENCODER_SAVE_ROOT}/${source_lang}-${target_lang}
INPUT_DIR=$DATA_DIR/${source_lang}-${target_lang}-tatoeba

mkdir -p $ENCODER_SAVE_DIR/${target_lang}
mkdir -p $ENCODER_SAVE_DIR/${source_lang}
mkdir -p $SAVE_DIR

## Save encoder outputs

# Save encoder outputs for source sentences
python $SAVE_ENCODER \
  ${INPUT_DIR} \
  --path ${MODEL} \
  --task translation_multi_simple_epoch \
  --lang-pairs ${source_lang}-${target_lang} \
  --lang-dict ${LANG_DICT} \
  --gen-subset ${SPLIT} \
  --bpe 'sentencepiece' \
  -s ${source_lang} -t ${target_lang} \
  --sentencepiece-model ${SPM} \
  --remove-bpe 'sentencepiece' \
  --beam 1 \
  --lang-tok-style mbart \
  --encoder-save-dir ${ENCODER_SAVE_DIR}/${source_lang}

## Save encoder outputs for target sentences
python $SAVE_ENCODER \
  ${INPUT_DIR} \
  --path ${MODEL} \
  --lang-pairs ${source_lang}-${target_lang} \
  --lang-dict ${LANG_DICT} \
  --task translation_multi_simple_epoch \
  --gen-subset ${SPLIT} \
  --bpe 'sentencepiece' \
  -t ${source_lang} -s ${target_lang} \
  --sentencepiece-model ${SPM} \
  --remove-bpe 'sentencepiece' \
  --beam 1 \
  --lang-tok-style mbart \
  --encoder-save-dir ${ENCODER_SAVE_DIR}/${target_lang}

## Mining
python mining/mine.py \
  --src-lang ${source_lang} \
  --tgt-lang ${target_lang} \
  --dim 1024 \
  --mem 10 \
  --neighborhood 4 \
  --src-dir ${ENCODER_SAVE_DIR}/${source_lang} \
  --tgt-dir ${ENCODER_SAVE_DIR}/${target_lang} \
  --output $SAVE_DIR \
  --threshold ${THRESHOLD} \
  --min-count ${MIN_COUNT} \
  --valid-size 100 \
  --dict-path ${DICT} \
  --spm-path ${SPM} \


## Process and binarize mined data
python $SPM_ENCODE \
  --model ${SPM} \
  --output_format=piece \
  --inputs mining/${source_lang}_${target_lang}_mined/train.${source_lang} mining/${source_lang}_${target_lang}_mined/train.${target_lang} \
  --outputs mining/${source_lang}_${target_lang}_mined/train.bpe.${source_lang} mining/${source_lang}_${target_lang}_mined/train.bpe.${target_lang}

python $SPM_ENCODE \
  --model ${SPM} \
  --output_format=piece \
  --inputs mining/${source_lang}_${target_lang}_mined/valid.${source_lang} mining/${source_lang}_${target_lang}_mined/valid.${target_lang} \
  --outputs mining/${source_lang}_${target_lang}_mined/valid.bpe.${source_lang} mining/${source_lang}_${target_lang}_mined/valid.bpe.${target_lang}


fairseq-preprocess \
  --source-lang ${source_lang} \
  --target-lang ${target_lang} \
  --trainpref mining/${source_lang}_${target_lang}_mined/train.bpe \
  --validpref mining/${source_lang}_${target_lang}_mined/valid.bpe \
  --destdir mining/${source_lang}_${target_lang}_mined \
  --srcdict ${DICT} \
  --joined-dictionary \
  --workers 8
