#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

SPM_ENCODE=flores/scripts/spm_encode.py
DATA=data_tmp
SPM_MODEL=criss_checkpoints/sentence.bpe.model
DICT=criss_checkpoints/dict.txt

git clone https://github.com/facebookresearch/LASER
mkdir -p data_tmp
declare -A lang_tatoeba_map=( ["ar_AR"]="ara" ["de_DE"]="deu"  ["es_XX"]="spa" ["et_EE"]="est" ["fi_FI"]="fin" ["fr_XX"]="fra" ["hi_IN"]="hin" ["it_IT"]="ita" ["ja_XX"]="jpn" ["ko_KR"]="kor" ["kk_KZ"]="kaz" ["nl_XX"]="nld" ["ru_RU"]="rus" ["tr_TR"]="tur" ["vi_VN"]="vie" ["zh_CN"]="cmn")
for lang in ar_AR de_DE es_XX et_EE fi_FI fr_XX hi_IN it_IT ja_XX kk_KZ ko_KR nl_XX ru_RU tr_TR vi_VN zh_CN; do
  lang_tatoeba=${lang_tatoeba_map[$lang]}
  echo $lang_tatoeba
  datadir=$DATA/${lang}-en_XX-tatoeba
  rm -rf $datadir
  mkdir -p $datadir
  TEST_PREFIX=LASER/data/tatoeba/v1/tatoeba
  python $SPM_ENCODE \
    --model ${SPM_MODEL} \
    --output_format=piece \
    --inputs ${TEST_PREFIX}.${lang_tatoeba}-eng.${lang_tatoeba} ${TEST_PREFIX}.${lang_tatoeba}-eng.eng \
    --outputs $datadir/test.bpe.${lang}-en_XX.${lang} $datadir/test.bpe.${lang}-en_XX.en_XX

  # binarize data
  fairseq-preprocess \
    --source-lang ${lang} --target-lang en_XX \
    --testpref $datadir/test.bpe.${lang}-en_XX \
    --destdir $datadir \
    --srcdict ${DICT} \
    --joined-dictionary \
    --workers 4
done
