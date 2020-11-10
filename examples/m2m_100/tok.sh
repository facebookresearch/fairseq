#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

TOKENIZERS_SCRIPTS=tokenizers
INSTALL_PATH=$TOKENIZERS_SCRIPTS/thirdparty

N_THREADS=8

lg=$1

MOSES=$INSTALL_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl

# special tokenization for Romanian
WMT16_SCRIPTS=$INSTALL_PATH/wmt16-scripts

NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py

# Burmese
MY_SEGMENT=$INSTALL_PATH/seg_my.py

# Arabic
AR_TOKENIZER=$TOKENIZERS_SCRIPTS/tokenizer_ar.sh

# Korean
KO_SEGMENT=$TOKENIZERS_SCRIPTS/seg_ko.sh

# Japanese
JA_SEGMENT=$TOKENIZERS_SCRIPTS/seg_ja.sh

# Indic
IN_TOKENIZER=$TOKENIZERS_SCRIPTS/tokenize_indic.py
INDIC_RESOURCES_PATH=$INSTALL_PATH/indic_nlp_resources

# Thai
THAI_TOKENIZER=$TOKENIZERS_SCRIPTS/tokenize_thai.py

# Chinese
CHINESE_TOKENIZER=$TOKENIZERS_SCRIPTS/tokenize_zh.py

# Chinese
if [ "$lg" = "zh" ]; then
  cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | python $CHINESE_TOKENIZER
# Thai
elif [ "$lg" = "th" ]; then
  cat - | python $THAI_TOKENIZER
# Japanese
elif [ "$lg" = "ja" ]; then
  cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | ${JA_SEGMENT}
# Korean
elif [ "$lg" = "ko" ]; then
  cat - | $REM_NON_PRINT_CHAR | ${KO_SEGMENT}
# Romanian
elif [ "$lg" = "ro" ]; then
  cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -no-escape -threads $N_THREADS -l $lg
# Burmese
elif [ "$lg" = "my" ]; then
  cat - | python ${MY_SEGMENT}
# Arabic
elif [ "$lg" = "ar" ]; then
  cat - | ${AR_TOKENIZER}
# Indic
elif [ "$lg" = "ne" ]; then
  cat - | python ${IN_TOKENIZER} $lg
elif [ "$lg" = "si" ]; then
  cat - | python ${IN_TOKENIZER} $lg
elif [ "$lg" = "hi" ]; then
  cat - | python ${IN_TOKENIZER} $lg
# other languages
else
  cat - | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $lg | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape -threads $N_THREADS -l $lg
fi
