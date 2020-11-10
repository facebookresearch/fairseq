#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
SRC=si_LK
TGT=en_XX
MODEL=criss_checkpoints/criss.3rd.pt

MULTIBLEU=mosesdecoder/scripts/generic/multi-bleu.perl
MOSES=mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
GEN_TMP_DIR=gen_tmp
LANG_DICT=criss_checkpoints/lang_dict.txt

if [ ! -d "mosesdecoder" ]; then
  git clone https://github.com/moses-smt/mosesdecoder
fi
mkdir -p $GEN_TMP_DIR
fairseq-generate data_tmp/${SRC}-${TGT}-flores \
        --task translation_multi_simple_epoch \
        --max-tokens 2000 \
        --path ${MODEL} \
        --skip-invalid-size-inputs-valid-test \
        --beam 5 --lenpen 1.0 --gen-subset test  \
        --remove-bpe=sentencepiece \
        --source-lang ${SRC} --target-lang ${TGT} \
        --decoder-langtok --lang-pairs 'en_XX-ar_AR,en_XX-de_DE,en_XX-es_XX,en_XX-fr_XX,en_XX-hi_IN,en_XX-it_IT,en_XX-ja_XX,en_XX-ko_KR,en_XX-nl_XX,en_XX-ru_RU,en_XX-zh_CN,en_XX-tr_TR,en_XX-vi_VN,en_XX-ro_RO,en_XX-my_MM,en_XX-ne_NP,en_XX-si_LK,en_XX-cs_CZ,en_XX-lt_LT,en_XX-kk_KZ,en_XX-gu_IN,en_XX-fi_FI,en_XX-et_EE,en_XX-lv_LV,ar_AR-en_XX,cs_CZ-en_XX,de_DE-en_XX,es_XX-en_XX,et_EE-en_XX,fi_FI-en_XX,fr_XX-en_XX,gu_IN-en_XX,hi_IN-en_XX,it_IT-en_XX,ja_XX-en_XX,kk_KZ-en_XX,ko_KR-en_XX,lt_LT-en_XX,lv_LV-en_XX,my_MM-en_XX,ne_NP-en_XX,nl_XX-en_XX,ro_RO-en_XX,ru_RU-en_XX,si_LK-en_XX,tr_TR-en_XX,vi_VN-en_XX,zh_CN-en_XX,ar_AR-es_XX,es_XX-ar_AR,ar_AR-hi_IN,hi_IN-ar_AR,ar_AR-zh_CN,zh_CN-ar_AR,cs_CZ-es_XX,es_XX-cs_CZ,cs_CZ-hi_IN,hi_IN-cs_CZ,cs_CZ-zh_CN,zh_CN-cs_CZ,de_DE-es_XX,es_XX-de_DE,de_DE-hi_IN,hi_IN-de_DE,de_DE-zh_CN,zh_CN-de_DE,es_XX-hi_IN,hi_IN-es_XX,es_XX-zh_CN,zh_CN-es_XX,et_EE-es_XX,es_XX-et_EE,et_EE-hi_IN,hi_IN-et_EE,et_EE-zh_CN,zh_CN-et_EE,fi_FI-es_XX,es_XX-fi_FI,fi_FI-hi_IN,hi_IN-fi_FI,fi_FI-zh_CN,zh_CN-fi_FI,fr_XX-es_XX,es_XX-fr_XX,fr_XX-hi_IN,hi_IN-fr_XX,fr_XX-zh_CN,zh_CN-fr_XX,gu_IN-es_XX,es_XX-gu_IN,gu_IN-hi_IN,hi_IN-gu_IN,gu_IN-zh_CN,zh_CN-gu_IN,hi_IN-zh_CN,zh_CN-hi_IN,it_IT-es_XX,es_XX-it_IT,it_IT-hi_IN,hi_IN-it_IT,it_IT-zh_CN,zh_CN-it_IT,ja_XX-es_XX,es_XX-ja_XX,ja_XX-hi_IN,hi_IN-ja_XX,ja_XX-zh_CN,zh_CN-ja_XX,kk_KZ-es_XX,es_XX-kk_KZ,kk_KZ-hi_IN,hi_IN-kk_KZ,kk_KZ-zh_CN,zh_CN-kk_KZ,ko_KR-es_XX,es_XX-ko_KR,ko_KR-hi_IN,hi_IN-ko_KR,ko_KR-zh_CN,zh_CN-ko_KR,lt_LT-es_XX,es_XX-lt_LT,lt_LT-hi_IN,hi_IN-lt_LT,lt_LT-zh_CN,zh_CN-lt_LT,lv_LV-es_XX,es_XX-lv_LV,lv_LV-hi_IN,hi_IN-lv_LV,lv_LV-zh_CN,zh_CN-lv_LV,my_MM-es_XX,es_XX-my_MM,my_MM-hi_IN,hi_IN-my_MM,my_MM-zh_CN,zh_CN-my_MM,ne_NP-es_XX,es_XX-ne_NP,ne_NP-hi_IN,hi_IN-ne_NP,ne_NP-zh_CN,zh_CN-ne_NP,nl_XX-es_XX,es_XX-nl_XX,nl_XX-hi_IN,hi_IN-nl_XX,nl_XX-zh_CN,zh_CN-nl_XX,ro_RO-es_XX,es_XX-ro_RO,ro_RO-hi_IN,hi_IN-ro_RO,ro_RO-zh_CN,zh_CN-ro_RO,ru_RU-es_XX,es_XX-ru_RU,ru_RU-hi_IN,hi_IN-ru_RU,ru_RU-zh_CN,zh_CN-ru_RU,si_LK-es_XX,es_XX-si_LK,si_LK-hi_IN,hi_IN-si_LK,si_LK-zh_CN,zh_CN-si_LK,tr_TR-es_XX,es_XX-tr_TR,tr_TR-hi_IN,hi_IN-tr_TR,tr_TR-zh_CN,zh_CN-tr_TR,vi_VN-es_XX,es_XX-vi_VN,vi_VN-hi_IN,hi_IN-vi_VN,vi_VN-zh_CN,zh_CN-vi_VN' \
        --lang-dict ${LANG_DICT} --lang-tok-style 'mbart' --sampling-method 'temperature' --sampling-temperature '1.0'  > $GEN_TMP_DIR/${SRC}_${TGT}.gen
cat $GEN_TMP_DIR/${SRC}_${TGT}.gen | grep -P "^T-" | cut -f2 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${TGT:0:2} | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape ${TGT:0:2} > $GEN_TMP_DIR/${SRC}_${TGT}.hyp
cat $GEN_TMP_DIR/${SRC}_${TGT}.gen | grep -P "^H-" | cut -f3 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${TGT:0:2} | $REM_NON_PRINT_CHAR | $TOKENIZER -no-escape ${TGT:0:2} > $GEN_TMP_DIR/${SRC}_${TGT}.ref
${MULTIBLEU} $GEN_TMP_DIR/${SRC}_${TGT}.ref < $GEN_TMP_DIR/${SRC}_${TGT}.hyp
