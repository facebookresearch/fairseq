#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if [ -z $WORKDIR_ROOT ] ;
then
        echo "please specify your working directory root in environment variable WORKDIR_ROOT. Exitting..."
        exit
fi



set -x -e

# TODO update the workdir and dest dir name
# put fasttext model
WORKDIR=$WORKDIR_ROOT
# put intermediate files
TMP_DIR=$WORKDIR_ROOT/tmp/tmp_wmt20_lowres_download
# output {train,valid,test} files to dest
DEST=$WORKDIR_ROOT/ML50/raw

UTILS=$PWD/utils

# per dataset locations
COMMONCRAWL_DIR=$TMP_DIR/commoncrawl
YANDEX_CORPUS=$WORKDIR_ROOT/wmt20/official/ru/yandex/1mcorpus.zip
# unzipped
CZENG_CORPUS=$WORKDIR_ROOT/wmt20/official/cs/czeng/czeng20-train
CCMT_DIR=$WORKDIR_ROOT/wmt20/official/zh/ccmt/parallel

download_and_select() {
  SUBFOLDER=$1
  URL=$2
  UNCOMPRESS_CMD=$3
  LANG=$4
  INPUT_FILEPATH=$5
  if [[ $# -gt 5 ]]; then
    LANG_COL=$6
    EN_COL=$7
  fi

  mkdir -p $SUBFOLDER
  cd $SUBFOLDER
  wget -nc --content-disposition $URL
  $UNCOMPRESS_CMD

  if [[ $# -gt 5 ]]; then
    cut -f$LANG_COL $INPUT_FILEPATH > $INPUT_FILEPATH.$LANG
    cut -f$EN_COL $INPUT_FILEPATH > $INPUT_FILEPATH.en
  fi
  cd ..

  ln -sf $SUBFOLDER/$INPUT_FILEPATH.$LANG $SUBFOLDER.$LANG
  ln -sf $SUBFOLDER/$INPUT_FILEPATH.en $SUBFOLDER.en
}

prepare_lid() {
  pip install fasttext

  # TODO specify global workdir
  MODEL=$WORKDIR/fasttext/lid.176.bin
  LID_MULTI=$UTILS/fasttext_multi_filter.py

  if [ ! -f "$MODEL" ]; then
    echo "downloading fasttext lid model..."
    mkdir -p $WORKDIR/fasttext
    wget -nc https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O $MODEL
  fi
}

prepare_moses() {
  pushd $UTILS
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git  
  popd
}

lid_filter() {
  # TODO specify global workdir
  MODEL=$WORKDIR/fasttext/lid.176.bin
  LID_MULTI=$UTILS/fasttext_multi_filter.py

  prepare_lid

  SRC=$1
  SRC_FILE=$2
  SRC_OUTPUT=$3
  TGT=$4
  TGT_FILE=$5
  TGT_OUTPUT=$6
  python $LID_MULTI --model $MODEL --inputs $SRC_FILE $TGT_FILE --langs $SRC $TGT --outputs $SRC_OUTPUT $TGT_OUTPUT
}

prepare_ja_ted() {
  mkdir -p ted
  cd ted

  wget -nc https://wit3.fbk.eu/archive/2017-01-trnted//texts/en/ja/en-ja.tgz
  tar -zxvf en-ja.tgz
  cat en-ja/train.tags.en-ja.en | grep -v -P "^[ ]*\<" | sed 's/^[ \t]*//g' | sed 's/[ \t]*$//g' > en-ja/train.en-ja.en
  cat en-ja/train.tags.en-ja.ja | grep -v -P "^[ ]*\<" | sed 's/^[ \t]*//g' | sed 's/[ \t]*$//g' > en-ja/train.en-ja.ja

  cd ..
  ln -sf ted/en-ja/train.en-ja.ja ted.ja
  ln -sf ted/en-ja/train.en-ja.en ted.en
}

prepare_ja() {
  OUTPUT_DIR=$TMP_DIR/ja
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  download_and_select paracrawl "http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/2.0/bitext/en-ja.tar.gz" "tar -zxvf en-ja.tar.gz" ja en-ja/en-ja.bicleaner05.txt 4 3 &
  download_and_select newscommentary "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-ja.tsv.gz" "gunzip -f news-commentary-v15.en-ja.tsv.gz" ja news-commentary-v15.en-ja.tsv 2 1 &
  download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.ja-en.tsv.gz" "gunzip -f wikititles-v2.ja-en.tsv.gz" ja wikititles-v2.ja-en.tsv 1 2 &
  download_and_select wikimatrix "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-ja.langid.tsv.gz" "gunzip -f WikiMatrix.v1.en-ja.langid.tsv.gz" ja WikiMatrix.v1.en-ja.langid.tsv 3 2 &
  download_and_select subtitle "https://nlp.stanford.edu/projects/jesc/data/split.tar.gz" "tar -zxvf split.tar.gz" ja split/train 2 1 &
  download_and_select kftt "http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz" "tar -zxvf kftt-data-1.0.tar.gz" ja kftt-data-1.0/data/orig/kyoto-train &

  prepare_ja_ted &

  # ted data needs to 

  wait

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.ja" | sort -V | xargs cat > all.ja
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  lid_filter ja all.ja $DEST/train.ja_XX-en_XX.ja_XX en all.en $DEST/train.ja_XX-en_XX.en_XX
}

prepare_ta() {
  OUTPUT_DIR=$TMP_DIR/ta
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.ta-en.tsv.gz" "gunzip -f wikititles-v2.ta-en.tsv.gz" ta wikititles-v2.ta-en.tsv 1 2 &
  download_and_select wikimatrix "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-ta.langid.tsv.gz" "gunzip -f WikiMatrix.v1.en-ta.langid.tsv.gz" ta WikiMatrix.v1.en-ta.langid.tsv 3 2 &
  download_and_select pmindia "http://data.statmt.org/pmindia/v1/parallel/pmindia.v1.ta-en.tsv" "" ta pmindia.v1.ta-en.tsv 2 1 &
  download_and_select tanzil "https://object.pouta.csc.fi/OPUS-Tanzil/v1/moses/en-ta.txt.zip" "unzip en-ta.txt.zip" ta Tanzil.en-ta &
  download_and_select pib "http://preon.iiit.ac.in/~jerin/resources/datasets/pib-v0.tar" "tar -xvf pib-v0.tar" ta pib/en-ta/train &
  download_and_select mkb "http://preon.iiit.ac.in/~jerin/resources/datasets/mkb-v0.tar" "tar -xvf mkb-v0.tar" ta mkb/en-ta/mkb &
  download_and_select ufal "http://ufal.mff.cuni.cz/~ramasamy/parallel/data/v2/en-ta-parallel-v2.tar.gz" "tar -zxvf en-ta-parallel-v2.tar.gz" ta en-ta-parallel-v2/corpus.bcn.train &

  wait

  # need special handling for nlpc
  mkdir -p nlpc
  cd nlpc
  wget -nc https://raw.githubusercontent.com/nlpc-uom/English-Tamil-Parallel-Corpus/master/En-Ta%20Corpus/En-Ta%20English.txt
  wget -nc https://github.com/nlpc-uom/English-Tamil-Parallel-Corpus/raw/master/En-Ta%20Corpus/En-Ta%20Tamil.txt
  tail -n +4 "En-Ta English.txt" > en-ta.en
  tail -n +4 "En-Ta Tamil.txt" > en-ta.ta
  cd ..
  ln -sf nlpc/en-ta.en nlpc.en
  ln -sf nlpc/en-ta.ta nlpc.ta

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.ta" | sort -V | xargs cat > all.ta
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  lid_filter ta all.ta $DEST/train.ta_IN-en_XX.ta_IN en all.en $DEST/train.ta_IN-en_XX.en_XX
}

prepare_iu() {
  OUTPUT_DIR=$TMP_DIR/iu
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR
  
  download_and_select nh "https://nrc-digital-repository.canada.ca/eng/view/dataset/?id=c7e34fa7-7629-43c2-bd6d-19b32bf64f60" "tar -zxvf Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0.1.tgz" iu Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/NunavutHansard > /dev/null &
  download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.iu-en.tsv.gz" "gunzip -f wikititles-v2.iu-en.tsv.gz" iu wikititles-v2.iu-en.tsv 1 2 &

  wait

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.iu" | sort -V | xargs cat | nh/Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0/scripts/normalize-iu-spelling.pl > all.iu
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  paste all.iu all.en | awk -F $'\t' '$1!=""&&$2!=""' > all.iuen
  cut -f1 all.iuen > $DEST/train.iu_CA-en_XX.iu_CA
  cut -f2 all.iuen > $DEST/train.iu_CA-en_XX.en_XX
}

prepare_km() {
  OUTPUT_DIR=$TMP_DIR/km
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  download_and_select paracrawl "http://data.statmt.org/wmt20/translation-task/ps-km/wmt20-sent.en-km.xz" "unxz wmt20-sent.en-km.zx" km wmt20-sent.en-km 2 1 &

  # km-parallel has multiple sets, concat all of them together
  mkdir -p opus
  cd opus
  wget -nc "http://data.statmt.org/wmt20/translation-task/ps-km/km-parallel.tgz"
  tar -zxvf km-parallel.tgz
  find ./km-parallel -maxdepth 1 -name "*.km" | sort -V | xargs cat > opus.km
  find ./km-parallel -maxdepth 1 -name "*.en" | sort -V | xargs cat > opus.en
  cd ..
  ln -sf opus/opus.km .
  ln -sf opus/opus.en .

  wait

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.km" | sort -V | xargs cat > all.km
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  lid_filter km all.km $DEST/train.km_KH-en_XX.km_KH en all.en $DEST/train.km_KH-en_XX.en_XX
}

prepare_ps() {
  OUTPUT_DIR=$TMP_DIR/ps
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  download_and_select paracrawl "http://data.statmt.org/wmt20/translation-task/ps-km/wmt20-sent.en-ps.xz" "unxz wmt20-sent.en-ps.xz" ps wmt20-sent.en-ps 2 1 &
  download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.ps-en.tsv.gz" "gunzip -f wikititles-v2.ps-en.tsv.gz" ps wikititles-v2.ps-en.tsv 1 2 &
  # ps-parallel has multiple sets, concat all of them together
  mkdir -p opus
  cd opus
  wget -nc "http://data.statmt.org/wmt20/translation-task/ps-km/ps-parallel.tgz"
  tar -zxvf ps-parallel.tgz
  find ./ps-parallel -maxdepth 1 -name "*.ps" | sort -V | xargs cat > opus.ps
  find ./ps-parallel -maxdepth 1 -name "*.en" | sort -V | xargs cat > opus.en
  cd ..
  ln -sf opus/opus.ps opus.ps
  ln -sf opus/opus.en opus.en

  wait

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.ps" | sort -V | xargs cat > all.ps
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  lid_filter ps all.ps $DEST/train.ps_AF-en_XX.ps_AF en all.en $DEST/train.ps_AF-en_XX.en_XX
}

download_commoncrawl() {
  mkdir -p $COMMONCRAWL_DIR
  cd $COMMONCRAWL_DIR

  wget -nc "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
  tar -zxvf training-parallel-commoncrawl.tgz
}
link_commoncrawl() {
  LANG=$1
  ln -sf $COMMONCRAWL_DIR/commoncrawl.$LANG-en.en commoncrawl.en
  ln -sf $COMMONCRAWL_DIR/commoncrawl.$LANG-en.$LANG commoncrawl.$LANG
}

strip_xlf() {
  INPUT_FILE=$1
  SRC=$2
  TGT=$3
  grep '<source xml:lang=' $INPUT_FILE | sed 's/^<[^<>]*>//g' | sed 's/<[^<>]*>$//g' > $INPUT_FILE.$SRC
  grep '<target xml:lang=' $INPUT_FILE | sed 's/^<[^<>]*>//g' | sed 's/<[^<>]*>$//g' > $INPUT_FILE.$TGT
}

download_and_process_tilde() {
  URL=$1
  UNCOMPRESS_CMD=$2
  FILENAME=$3
  LANG=$4
  PROCESS_CMD=$5

  mkdir -p tilde
  cd tilde
  wget -nc $URL
  $UNCOMPRESS_CMD
  echo "executing cmd"
  echo $PROCESS_CMD
  $PROCESS_CMD
  cd ..
  ln -sf tilde/$FILENAME.$LANG tilde.$LANG
  ln -sf tilde/$FILENAME.en tilde.en
}

prepare_cs() {
  OUTPUT_DIR=$TMP_DIR/cs
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  #download_and_select europarl "http://www.statmt.org/europarl/v10/training/europarl-v10.cs-en.tsv.gz" "gunzip europarl-v10.cs-en.tsv.gz" cs europarl-v10.cs-en.tsv 1 2 &
  #download_and_select paracrawl "https://s3.amazonaws.com/web-language-models/paracrawl/release5.1/en-cs.txt.gz" "gunzip en-cs.txt.gz" cs en-cs.txt 2 1 &
  #link_commoncrawl cs
  #download_and_select newscommentary "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.cs-en.tsv.gz" "gunzip news-commentary-v15.cs-en.tsv.gz" cs news-commentary-v15.cs-en.tsv 1 2 &
  #download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.cs-en.tsv.gz" "gunzip wikititles-v2.cs-en.tsv.gz" cs wikititles-v2.cs-en.tsv 1 2 &
  #download_and_process_tilde "http://data.statmt.org/wmt20/translation-task/rapid/RAPID_2019.cs-en.xlf.gz" "gunzip RAPID_2019.cs-en.xlf.gz" RAPID_2019.cs-en.xlf cs "strip_xlf RAPID_2019.cs-en.xlf cs en" &
  #download_and_select wikimatrix "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.cs-en.langid.tsv.gz" "gunzip WikiMatrix.v1.cs-en.langid.tsv.gz" cs WikiMatrix.v1.cs-en.langid.tsv 2 3 &

  #wait

  # remove previous results
  #rm -f all.??
  #find ./ -maxdepth 1 -name "*.cs" | sort -V | xargs cat > all.cs
  #find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  if [ -z $CZENG_CORPUS ] ;
  then
          echo "Please download CZENG_CORPUS manually and place them at $CZENG_CORPUS. Exitting..."
          exit
  fi  
  cat $CZENG_CORPUS | sed '/^$/d' | cut -f5 > all.cs
  cat $CZENG_CORPUS | sed '/^$/d' | cut -f6 > all.en

  lid_filter cs all.cs $DEST/train.cs_CZ-en_XX.cs_CZ en all.en $DEST/train.cs_CZ-en_XX.en_XX
}

prepare_de() {
  OUTPUT_DIR=$TMP_DIR/de
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  download_and_select europarl "http://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz" "gunzip europarl-v10.de-en.tsv.gz" de europarl-v10.de-en.tsv 1 2 &
  download_and_select paracrawl "https://s3.amazonaws.com/web-language-models/paracrawl/release5.1/en-de.txt.gz"  "gunzip en-de.txt.gz" de en-de.txt 2 1 &
  link_commoncrawl de
  download_and_select newscommentary "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.de-en.tsv.gz" "gunzip news-commentary-v15.de-en.tsv.gz" de news-commentary-v15.de-en.tsv 1 2 &
  download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.de-en.tsv.gz" "gunzip wikititles-v2.de-en.tsv.gz" de wikititles-v2.de-en.tsv 1 2 &
  download_and_process_tilde "http://data.statmt.org/wmt20/translation-task/rapid/RAPID_2019.de-en.xlf.gz" "gunzip RAPID_2019.de-en.xlf.gz" RAPID_2019.de-en.xlf de "strip_xlf RAPID_2019.de-en.xlf de en" &
  download_and_select wikimatrix "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.de-en.langid.tsv.gz" "gunzip WikiMatrix.v1.de-en.langid.tsv.gz" de WikiMatrix.v1.de-en.langid.tsv 2 3 &

  wait

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.de" | sort -V | xargs cat > all.de
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  lid_filter de all.de $DEST/train.de_DE-en_XX.de_DE en all.en $DEST/train.de_DE-en_XX.en_XX
}

prepare_tmx() {
  TMX_FILE=$1
  git clone https://github.com/amake/TMX2Corpus $UTILS/tmx2corpus
  pip install tinysegmenter

  python $UTILS/tmx2corpus/tmx2corpus.py $TMX_FILE
}

prepare_pl() {
  OUTPUT_DIR=$TMP_DIR/pl
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  # download_and_select europarl "http://www.statmt.org/europarl/v10/training/europarl-v10.pl-en.tsv.gz" "gunzip europarl-v10.pl-en.tsv.gz" pl europarl-v10.pl-en.tsv 1 2 &
  # download_and_select paracrawl "https://s3.amazonaws.com/web-language-models/paracrawl/release5.1/en-pl.txt.gz" "gunzip en-pl.txt.gz" pl en-pl.txt 2 1 &
  # download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.pl-en.tsv.gz" "gunzip wikititles-v2.pl-en.tsv.gz" pl wikititles-v2.pl-en.tsv 1 2 &
  download_and_select tilde "https://tilde-model.s3-eu-west-1.amazonaws.com/rapid2019.en-pl.tmx.zip" "gunzip rapid2019.en-pl.tmx.zip" bitext pl "prepare_tmx RAPID_2019.UNIQUE.en-pl.tmx" &
  # download_and_select wikimatrix "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-pl.langid.tsv.gz" "gunzip WikiMatrix.v1.en-pl.langid.tsv.gz" pl WikiMatrix.v1.en-pl.langid.tsv 3 2 &

  wait

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.pl" | sort -V | xargs cat > all.pl
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  lid_filter pl all.pl $DEST/train.pl_PL-en_XX.pl_PL en all.en $DEST/train.pl_PL-en_XX.en_XX
}

prepare_uncorpus() {
  $URLS=$1
  $FILES=$2

  mkdir -p uncorpus
  cd uncorpus

  for URL in $URLS; do
    wget -nc $URL
  done
  cat $FILES > uncorpus.tar.gz
  tar -zxvf uncorpus.tar.gz

  cd ..
  ln -sf uncorpus/en-$LANG/UNv1.0.en-$LANG.$LANG uncorpus.$LANG
  ln -sf uncorpus/en-$LANG/UNv1.0.en-$LANG.en uncorpus.en
}

prepare_yandex() {
  mkdir -p yandex
  cd yandex
  unzip $YANDEX_CORPUS ./
  cd ..
  ln -s yandex/corpus.en_ru.1m.en yandex.en
  ln -s yandex/corpus.en_ru.1m.ru yandex.ru
}

prepare_ru() {
  OUTPUT_DIR=$TMP_DIR/ru
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  download_and_select paracrawl "https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz" "tar -zxvf paracrawl-release1.en-ru.zipporah0-dedup-clean.tgz" ru paracrawl-release1.en-ru.zipporah0-dedup-clean &
  link_commoncrawl ru
  download_and_select newscommentary "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-ru.tsv.gz" "gunzip news-commentary-v15.en-ru.tsv.gz" ru news-commentary-v15.en-ru.tsv 2 1 &
  prepare_yandex &
  download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.ru-en.tsv.gz" "gunzip wikititles-v2.ru-en.tsv.gz" ru wikititles-v2.ru-en.tsv 1 2 &
  prepare_uncorpus "https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-ru.tar.gz.00 https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-ru.tar.gz.01 https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-ru.tar.gz.02" "UNv1.0.en-ru.tar.gz.00 UNv1.0.en-ru.tar.gz.01 UNv1.0.en-ru.tar.gz.02" &
  download_and_select wikimatrix "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-ru.langid.tsv.gz" "gunzip WikiMatrix.v1.en-ru.langid.tsv.gz" ru WikiMatrix.v1.en-ru.langid.tsv 3 2 &

  wait

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.ru" | sort -V | xargs cat > all.ru
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  lid_filter ru all.ru $DEST/train.ru_RU-en_XX.ru_RU en all.en $DEST/train.ru_RU-en_XX.en_XX
}

prepare_ccmt() {
  mkdir -p ccmt
  cd ccmt
  # assume ccmt data is already unzipped under CCMT_DIR folder
  cat $CCMT_DIR/datum2017/Book*_cn.txt | sed 's/ //g' > datum2017.detok.zh
  cat $CCMT_DIR/datum2017/Book*_en.txt > datum2017.detok.en
  cat $CCMT_DIR/casict2011/casict-A_ch.txt $CCMT_DIR/casict2011/casict-B_ch.txt $CCMT_DIR/casict2015/casict2015_ch.txt $CCMT_DIR/datum2015/datum_ch.txt $CCMT_DIR/neu2017/NEU_cn.txt datum2017.detok.zh > ccmt.zh
  cat $CCMT_DIR/casict2011/casict-A_en.txt $CCMT_DIR/casict2011/casict-B_en.txt $CCMT_DIR/casict2015/casict2015_en.txt $CCMT_DIR/datum2015/datum_en.txt $CCMT_DIR/neu2017/NEU_en.txt datum2017.detok.en > ccmt.en
  cd ..
  ln -sf ccmt/ccmt.zh ccmt.zh
  ln -sf ccmt/ccmt.en ccmt.en
}

prepare_zh() {
  OUTPUT_DIR=$TMP_DIR/zh
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR

  download_and_select newscommentary "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-zh.tsv.gz" "gunzip news-commentary-v15.en-zh.tsv.gz" zh news-commentary-v15.en-zh.tsv 2 1 &
  download_and_select wikititles "http://data.statmt.org/wikititles/v2/wikititles-v2.zh-en.tsv.gz" "gunzip wikititles-v2.zh-en.tsv.gz" zh wikititles-v2.zh-en.tsv 1 2 &
  prepare_uncorpus "https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-zh.tar.gz.00 https://stuncorpusprod.blob.core.windows.net/corpusfiles/UNv1.0.en-zh.tar.gz.01" "UNv1.0.en-zh.tar.gz.00 UNv1.0.en-zh.tar.gz.01" &
  prepare_ccmt &
  download_and_select wikimatrix "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-zh.langid.tsv.gz" "gunzip WikiMatrix.v1.en-zh.langid.tsv.gz" zh WikiMatrix.v1.en-zh.langid.tsv 3 2 &

  wait

  # remove previous results
  rm -f all.??
  find ./ -maxdepth 1 -name "*.zh" | sort -V | xargs cat > all.zh
  find ./ -maxdepth 1 -name "*.en" | sort -V | xargs cat > all.en
  lid_filter zh all.zh $DEST/train.zh_CN-en_XX.zh_CN en all.en $DEST/train.zh_CN-en_XX.en_XX
}

prepare_tests() {
  OUTPUT_DIR=$TMP_DIR
  mkdir -p $OUTPUT_DIR
  cd $OUTPUT_DIR
  wget -nc http://data.statmt.org/wmt20/translation-task/dev.tgz
  tar -zxvf dev.tgz
  cd dev

  cat newsdev2020-jaen-src.ja.sgm | $UTILS/strip_sgm.sh > newsdev2020-jaen.ja
  cat newsdev2020-jaen-ref.en.sgm | $UTILS/strip_sgm.sh > newsdev2020-jaen.en
  split newsdev2020-jaen.ja -a 0 -n r/1/2 > $DEST/valid.ja_XX-en_XX.ja_XX
  split newsdev2020-jaen.en -a 0 -n r/1/2 > $DEST/valid.ja_XX-en_XX.en_XX
  split newsdev2020-jaen.ja -a 0 -n r/2/2 > $DEST/test.ja_XX-en_XX.ja_XX
  split newsdev2020-jaen.en -a 0 -n r/2/2 > $DEST/test.ja_XX-en_XX.en_XX

  cat newsdev2020-iuen-src.iu.sgm | strip_sgm.sh > newsdev2020-iuen.iu
  cat newsdev2020-iuen-ref.en.sgm | strip_sgm.sh > newsdev2020-iuen.en
  split newsdev2020-iuen.iu -a 0 -n r/1/2 > $DEST/valid.iu_CA-en_XX.iu_CA
  split newsdev2020-iuen.en -a 0 -n r/1/2 > $DEST/valid.iu_CA-en_XX.en_XX
  split newsdev2020-iuen.iu -a 0 -n r/2/2 > $DEST/test.iu_CA-en_XX.iu_CA
  split newsdev2020-iuen.en -a 0 -n r/2/2 > $DEST/test.iu_CA-en_XX.en_XX

  cat newsdev2020-taen-src.ta.sgm | strip_sgm.sh > newsdev2020-taen.ta
  cat newsdev2020-taen-ref.en.sgm | strip_sgm.sh > newsdev2020-taen.en
  split newsdev2020-taen.ta -a 0 -n r/1/2 > $DEST/valid.ta_IN-en_XX.ta_IN
  split newsdev2020-taen.en -a 0 -n r/1/2 > $DEST/valid.ta_IN-en_XX.en_XX
  split newsdev2020-taen.ta -a 0 -n r/2/2 > $DEST/test.ta_IN-en_XX.ta_IN
  split newsdev2020-taen.en -a 0 -n r/2/2 > $DEST/test.ta_IN-en_XX.en_XX

  cp wikipedia.dev.km-en.km $DEST/valid.km_KH-en_XX.km_KH
  cp wikipedia.dev.km-en.en $DEST/valid.km_KH-en_XX.en_XX
  cp wikipedia.devtest.km-en.km $DEST/test.km_KH-en_XX.km_KH
  cp wikipedia.devtest.km-en.en $DEST/test.km_KH-en_XX.en_XX

  cp wikipedia.dev.ps-en.ps $DEST/valid.ps_AF-en_XX.ps_AF
  cp wikipedia.dev.ps-en.en $DEST/valid.ps_AF-en_XX.en_XX
  cp wikipedia.devtest.ps-en.ps $DEST/test.ps_AF-en_XX.ps_AF
  cp wikipedia.devtest.ps-en.en $DEST/test.ps_AF-en_XX.en_XX

  cat newsdev2020-plen-src.pl.sgm | strip_sgm.sh > newsdev2020-plen.pl
  cat newsdev2020-plen-ref.en.sgm | strip_sgm.sh > newsdev2020-plen.en
  split newsdev2020-plen.pl -a 0 -n r/1/2 > $DEST/valid.pl_PL-en_XX.pl_PL
  split newsdev2020-plen.en -a 0 -n r/1/2 > $DEST/valid.pl_PL-en_XX.en_XX
  split newsdev2020-plen.pl -a 0 -n r/2/2 > $DEST/test.pl_PL-en_XX.pl_PL
  split newsdev2020-plen.en -a 0 -n r/2/2 > $DEST/test.pl_PL-en_XX.en_XX

  cat newstest2018-encs-src.en.sgm | strip_sgm.sh > $DEST/valid.en_XX-cs_CZ.en_XX
  cat newstest2018-encs-ref.cs.sgm | strip_sgm.sh > $DEST/valid.en_XX-cs_CZ.cs_CZ
  cat newstest2019-encs-src.en.sgm | strip_sgm.sh > $DEST/test.en_XX-cs_CZ.en_XX
  cat newstest2019-encs-ref.cs.sgm | strip_sgm.sh > $DEST/test.en_XX-cs_CZ.cs_CZ

  cat newstest2018-deen-src.de.sgm | strip_sgm.sh > $DEST/valid.de_DE-en_XX.de_DE
  cat newstest2018-deen-ref.en.sgm | strip_sgm.sh > $DEST/valid.de_DE-en_XX.en_XX
  cat newstest2018-ende-src.en.sgm | strip_sgm.sh > $DEST/valid.en_XX-de_DE.en_XX
  cat newstest2018-ende-ref.de.sgm | strip_sgm.sh > $DEST/valid.en_XX-de_DE.de_DE
  cat newstest2019-deen-src.de.sgm | strip_sgm.sh > $DEST/test.de_DE-en_XX.de_DE
  cat newstest2019-deen-ref.en.sgm | strip_sgm.sh > $DEST/test.de_DE-en_XX.en_XX
  cat newstest2019-ende-src.en.sgm | strip_sgm.sh > $DEST/test.en_XX-de_DE.en_XX
  cat newstest2019-ende-ref.de.sgm | strip_sgm.sh > $DEST/test.en_XX-de_DE.de_DE

  cat newstest2018-ruen-src.ru.sgm | strip_sgm.sh > $DEST/valid.ru_RU-en_XX.ru_RU
  cat newstest2018-ruen-ref.en.sgm | strip_sgm.sh > $DEST/valid.ru_RU-en_XX.en_XX
  cat newstest2018-enru-src.en.sgm | strip_sgm.sh > $DEST/valid.en_XX-ru_RU.en_XX
  cat newstest2018-enru-ref.ru.sgm | strip_sgm.sh > $DEST/valid.en_XX-ru_RU.ru_RU
  cat newstest2019-ruen-src.ru.sgm | strip_sgm.sh > $DEST/test.ru_RU-en_XX.ru_RU
  cat newstest2019-ruen-ref.en.sgm | strip_sgm.sh > $DEST/test.ru_RU-en_XX.en_XX
  cat newstest2019-enru-src.en.sgm | strip_sgm.sh > $DEST/test.en_XX-ru_RU.en_XX
  cat newstest2019-enru-ref.ru.sgm | strip_sgm.sh > $DEST/test.en_XX-ru_RU.ru_RU

  cat newstest2018-zhen-src.zh.sgm | strip_sgm.sh > $DEST/valid.zh_CN-en_XX.zh_CN
  cat newstest2018-zhen-ref.en.sgm | strip_sgm.sh > $DEST/valid.zh_CN-en_XX.en_XX
  cat newstest2018-enzh-src.en.sgm | strip_sgm.sh > $DEST/valid.en_XX-zh_CN.en_XX
  cat newstest2018-enzh-ref.zh.sgm | strip_sgm.sh > $DEST/valid.en_XX-zh_CN.zh_CN
  cat newstest2019-zhen-src.zh.sgm | strip_sgm.sh > $DEST/test.zh_CN-en_XX.zh_CN
  cat newstest2019-zhen-ref.en.sgm | strip_sgm.sh > $DEST/test.zh_CN-en_XX.en_XX
  cat newstest2019-enzh-src.en.sgm | strip_sgm.sh > $DEST/test.en_XX-zh_CN.en_XX
  cat newstest2019-enzh-ref.zh.sgm | strip_sgm.sh > $DEST/test.en_XX-zh_CN.zh_CN
}

mkdir -p $DEST

prepare_lid
prepare_moses
download_commoncrawl

prepare_ja &
prepare_ta &
prepare_km &
prepare_ps &
prepare_iu &
prepare_cs &
prepare_de &
prepare_pl &
prepare_ru &
prepare_zh &

# prepare valid/test set
prepare_tests &

# wait

# TODO remove intermediate files
# rm -rf $TMP_DIR
