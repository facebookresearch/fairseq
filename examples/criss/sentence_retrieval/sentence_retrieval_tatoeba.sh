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
ENCODER_ANALYSIS=sentence_retrieval/encoder_analysis.py
SAVE_ENCODER=save_encoder.py
ENCODER_SAVE_ROOT=sentence_embeddings/$MODEL



DATA_DIR=data_tmp
INPUT_DIR=$DATA_DIR/${source_lang}-${target_lang}-tatoeba
ENCODER_SAVE_DIR=${ENCODER_SAVE_ROOT}/${source_lang}-${target_lang}
mkdir -p $ENCODER_SAVE_DIR/${target_lang}
mkdir -p $ENCODER_SAVE_DIR/${source_lang}

# Save encoder outputs for source sentences
python $SAVE_ENCODER \
  ${INPUT_DIR} \
  --path ${MODEL} \
  --task translation_multi_simple_epoch \
  --lang-dict ${LANG_DICT} \
  --gen-subset ${SPLIT} \
  --bpe 'sentencepiece' \
  --lang-pairs ${source_lang}-${target_lang} \
  -s ${source_lang} -t ${target_lang} \
  --sentencepiece-model ${SPM} \
  --remove-bpe 'sentencepiece' \
  --beam 1 \
  --lang-tok-style mbart \
  --encoder-save-dir ${ENCODER_SAVE_DIR}/${source_lang}

# Save encoder outputs for target sentences
python $SAVE_ENCODER \
  ${INPUT_DIR} \
  --path ${MODEL} \
  --lang-dict ${LANG_DICT} \
  --task translation_multi_simple_epoch \
  --gen-subset ${SPLIT} \
  --bpe 'sentencepiece' \
  --lang-pairs ${target_lang}-${source_lang} \
  -t ${source_lang} -s ${target_lang} \
  --sentencepiece-model ${SPM} \
  --remove-bpe 'sentencepiece' \
  --beam 1 \
  --lang-tok-style mbart \
  --encoder-save-dir ${ENCODER_SAVE_DIR}/${target_lang}

# Analyze sentence retrieval accuracy
python $ENCODER_ANALYSIS --langs "${source_lang},${target_lang}" ${ENCODER_SAVE_DIR}
