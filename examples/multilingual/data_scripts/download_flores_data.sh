#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

if [ -z $WORKDIR_ROOT ] ;
then
        echo "please specify your working directory root in environment variable WORKDIR_ROOT. Exitting..."
        exit
fi


set -e
set -o pipefail

SRC=en
SI_TGT=si
NE_TGT=ne

DESTDIR=${WORKDIR_ROOT}/ML50/raw/

ROOT=${WORKDIR_ROOT}/tmp
mkdir -p $ROOT
DATA=$ROOT/data
NE_ROOT=$DATA/all-clean-ne
SI_ROOT=$DATA/all-clean-si

mkdir -p $DATA $NE_ROOT $SI_ROOT

SI_OPUS_DATASETS=(
  "$SI_ROOT/GNOME.en-si"
  "$SI_ROOT/Ubuntu.en-si"
  "$SI_ROOT/KDE4.en-si"
  "$SI_ROOT/OpenSubtitles.en-si"
)

SI_OPUS_URLS=(
  "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en-si.txt.zip"
  "https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-si.txt.zip"
  "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-si.txt.zip"
  "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-si.txt.zip"
)

NE_OPUS_DATASETS=(
  "$NE_ROOT/GNOME.en-ne"
  "$NE_ROOT/Ubuntu.en-ne"
  "$NE_ROOT/KDE4.en-ne"
)

NE_OPUS_URLS=(
  "https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/en-ne.txt.zip"
  "https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/moses/en-ne.txt.zip"
  "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-ne.txt.zip"
)

REMOVE_FILE_PATHS=()

# Download data
download_data() {
  CORPORA=$1
  URL=$2

  if [ -f $CORPORA ]; then
    echo "$CORPORA already exists, skipping download"
  else
    echo "Downloading $URL"
    wget $URL -O $CORPORA --no-check-certificate || rm -f $CORPORA
    if [ -f $CORPORA ]; then
      echo "$URL successfully downloaded."
    else
      echo "$URL not successfully downloaded."
      rm -f $CORPORA
      exit -1
    fi
  fi
}

# Example: download_opus_data $LANG_ROOT $TGT
download_opus_data() {
  LANG_ROOT=$1
  TGT=$2

  if [ "$TGT" = "si" ]; then
    URLS=("${SI_OPUS_URLS[@]}")
    DATASETS=("${SI_OPUS_DATASETS[@]}")
  else
    URLS=("${NE_OPUS_URLS[@]}")
    DATASETS=("${NE_OPUS_DATASETS[@]}")
  fi

  # Download and extract data
  for ((i=0;i<${#URLS[@]};++i)); do
    URL=${URLS[i]}
    CORPORA=${DATASETS[i]}

    download_data $CORPORA $URL
    unzip -o $CORPORA -d $LANG_ROOT
    REMOVE_FILE_PATHS+=( $CORPORA $CORPORA.xml $CORPORA.ids $LANG_ROOT/README $LANG_ROOT/LICENSE )
  done

  cat ${DATASETS[0]}.$SRC ${DATASETS[1]}.$SRC ${DATASETS[2]}.$SRC > $LANG_ROOT/GNOMEKDEUbuntu.$SRC-$TGT.$SRC
  cat ${DATASETS[0]}.$TGT ${DATASETS[1]}.$TGT ${DATASETS[2]}.$TGT > $LANG_ROOT/GNOMEKDEUbuntu.$SRC-$TGT.$TGT

  REMOVE_FILE_PATHS+=( ${DATASETS[0]}.$SRC ${DATASETS[1]}.$SRC ${DATASETS[2]}.$SRC )
  REMOVE_FILE_PATHS+=( ${DATASETS[0]}.$TGT ${DATASETS[1]}.$TGT ${DATASETS[2]}.$TGT )
}

download_opus_data $SI_ROOT $SI_TGT
cp ${SI_OPUS_DATASETS[3]}.$SRC $SI_ROOT/OpenSubtitles2018.$SRC-$SI_TGT.$SRC
cp ${SI_OPUS_DATASETS[3]}.$SI_TGT $SI_ROOT/OpenSubtitles2018.$SRC-$SI_TGT.$SI_TGT
REMOVE_FILE_PATHS+=( ${SI_OPUS_DATASETS[3]}.$SRC ${SI_OPUS_DATASETS[3]}.$SI_TGT )

download_opus_data $NE_ROOT $NE_TGT


# Download and extract Global Voices data
GLOBAL_VOICES="$NE_ROOT/globalvoices.2018q4.ne-en"
GLOBAL_VOICES_URL="http://www.casmacat.eu/corpus/global-voices/globalvoices.ne-en.xliff.gz"

download_data $GLOBAL_VOICES.gz $GLOBAL_VOICES_URL
gunzip -Nf $GLOBAL_VOICES.gz

sed -ne 's?.*<source>\(.*\)</source>.*?\1?p' $GLOBAL_VOICES > $GLOBAL_VOICES.$NE_TGT
sed -ne 's?.*<target[^>]*>\(.*\)</target>.*?\1?p' $GLOBAL_VOICES > $GLOBAL_VOICES.$SRC

REMOVE_FILE_PATHS+=( $GLOBAL_VOICES )

# Download and extract the bible dataset
BIBLE_TOOLS=bible-corpus-tools
XML_BIBLES=XML_Bibles
XML_BIBLES_DUP=XML_Bibles_dup

if [ ! -e $BIBLE_TOOLS ]; then
    echo "Cloning bible-corpus-tools repository..."
    git clone https://github.com/christos-c/bible-corpus-tools.git
fi

mkdir -p $BIBLE_TOOLS/bin $XML_BIBLES $XML_BIBLES_DUP
javac -cp "$BIBLE_TOOLS/lib/*" -d $BIBLE_TOOLS/bin $BIBLE_TOOLS/src/bible/readers/*.java $BIBLE_TOOLS/src/bible/*.java

download_data bible.tar.gz "https://github.com/christos-c/bible-corpus/archive/v1.2.1.tar.gz"
tar xvzf bible.tar.gz

cp bible-corpus-1.2.1/bibles/{Greek.xml,English.xml,Nepali.xml} $XML_BIBLES/
cp bible-corpus-1.2.1/bibles/{Greek.xml,English-WEB.xml,Nepali.xml} $XML_BIBLES_DUP/

java -cp $BIBLE_TOOLS/lib/*:$BIBLE_TOOLS/bin bible.CreateMLBooks $XML_BIBLES
java -cp $BIBLE_TOOLS/lib/*:$BIBLE_TOOLS/bin bible.CreateMLBooks $XML_BIBLES_DUP
java -cp $BIBLE_TOOLS/lib/*:$BIBLE_TOOLS/bin bible.CreateVerseAlignedBooks $XML_BIBLES
java -cp $BIBLE_TOOLS/lib/*:$BIBLE_TOOLS/bin bible.CreateVerseAlignedBooks $XML_BIBLES_DUP

cat $XML_BIBLES/aligned/*/English.txt > $NE_ROOT/bible.$SRC-$NE_TGT.$SRC
cat $XML_BIBLES/aligned/*/Nepali.txt > $NE_ROOT/bible.$SRC-$NE_TGT.$NE_TGT
cat $XML_BIBLES_DUP/aligned/*/English-WEB.txt > $NE_ROOT/bible_dup.$SRC-$NE_TGT.$SRC
cat $XML_BIBLES_DUP/aligned/*/Nepali.txt > $NE_ROOT/bible_dup.$SRC-$NE_TGT.$NE_TGT
REMOVE_FILE_PATHS+=( bible-corpus-1.2.1 bible.tar.gz $BIBLE_TOOLS $XML_BIBLES $XML_BIBLES_DUP )

# Download and extract the Penn Treebank dataset
NE_TAGGED=$ROOT/new_submissions_parallel_corpus_project_Nepal
NE_TAGGED_URL="http://www.cle.org.pk/Downloads/ling_resources/parallelcorpus/NepaliTaggedCorpus.zip"
EN_TAGGED_PATCH_URL="https://dl.fbaipublicfiles.com/fairseq/data/nepali-penn-treebank.en.patch"
NE_TAGGED_PATCH_URL="https://dl.fbaipublicfiles.com/fairseq/data/nepali-penn-treebank.ne.patch"
MOSES=mosesdecoder
MOSES_TOK=$MOSES/scripts/tokenizer
EN_PATCH_REGEX="{s:\\\/:\/:g;s/\*\T\*\-\n+//g;s/\-LCB\-/\{/g;s/\-RCB\-/\}/g; s/\-LSB\-/\[/g; s/\-RSB\-/\]/g;s/\-LRB\-/\(/g; s/\-RRB\-/\)/g; s/\'\'/\"/g; s/\`\`/\"/g; s/\ +\'s\ +/\'s /g; s/\ +\'re\ +/\'re /g; s/\"\ +/\"/g; s/\ +\"/\"/g; s/\ n't([\ \.\"])/n't\1/g; s/\r+(.)/\1/g;}"
NE_PATCH_REGEX="{s:\p{Cf}::g;s:\\\/:\/:g;s/\*\T\*\-\n+//g;s/\-LCB\-/\{/g;s/\-RCB\-/\}/g; s/\-LSB\-/\[/g; s/\-RSB\-/\]/g;s/\-LRB\-/\(/g; s/\-RRB\-/\)/g; s/\'\'/\"/g; s/\`\`/\"/g; s/\ +\'s\ +/\'s /g; s/\ +\'re\ +/\'re /g; s/\"\ +/\"/g; s/\ +\"/\"/g; s/\ n't([\ \.\"])/n't\1/g; s/\r+(.)/\1/g;}"

download_data $DATA/nepali-penn-treebank.$SRC.patch $EN_TAGGED_PATCH_URL
download_data $DATA/nepali-penn-treebank.$NE_TGT.patch $NE_TAGGED_PATCH_URL
download_data original.zip $NE_TAGGED_URL
unzip -o original.zip -d $ROOT

cat $NE_TAGGED/00.txt $NE_TAGGED/01.txt $NE_TAGGED/02.txt > $NE_TAGGED/nepali-penn-treebank.$SRC
cat $NE_TAGGED/00ne_revised.txt $NE_TAGGED/01ne_revised.txt $NE_TAGGED/02ne_revised.txt > $NE_TAGGED/nepali-penn-treebank.$NE_TGT

patch $NE_TAGGED/nepali-penn-treebank.$SRC -i $DATA/nepali-penn-treebank.$SRC.patch -o $NE_TAGGED/nepali-penn-treebank-patched.$SRC
patch $NE_TAGGED/nepali-penn-treebank.$NE_TGT -i $DATA/nepali-penn-treebank.$NE_TGT.patch -o $NE_TAGGED/nepali-penn-treebank-patched.$NE_TGT

if [ ! -e $MOSES ]; then
    echo "Cloning moses repository..."
    git clone https://github.com/moses-smt/mosesdecoder.git
fi

cat $NE_TAGGED/nepali-penn-treebank-patched.$SRC | \
  perl -anpe "$EN_PATCH_REGEX"  | \
  $MOSES_TOK/tokenizer.perl -l $SRC | \
  $MOSES_TOK/detokenizer.perl -l $SRC > $NE_ROOT/nepali-penn-treebank.$SRC

cat $NE_TAGGED/nepali-penn-treebank-patched.$NE_TGT | \
  perl -CIO -anpe "$NE_PATCH_REGEX" | \
  $MOSES_TOK/detokenizer.perl -l $SRC > $NE_ROOT/nepali-penn-treebank.$NE_TGT


# Download nepali dictionary data
NE_DICT=$NE_ROOT/dictionaries
download_data $NE_DICT "http://www.seas.upenn.edu/~nlp/resources/TACL-data-release/dictionaries.tar.gz"
tar xvzf $NE_DICT
cp dictionaries/dict.ne $NE_ROOT/dictionary.$NE_TGT-$SRC
REMOVE_FILE_PATHS+=( $NE_DICT dictionaries )

REMOVE_FILE_PATHS+=( $MOSES $NE_TAGGED original.zip $DATA/nepali-penn-treebank.$SRC.patch $DATA/nepali-penn-treebank.$NE_TGT.patch )


# Remove the temporary files
for ((i=0;i<${#REMOVE_FILE_PATHS[@]};++i)); do
  rm -rf ${REMOVE_FILE_PATHS[i]}
done

# Copy the training data
si=si_LK
ne=ne_NP
en=en_XX
cat $SI_ROOT/GNOMEKDEUbuntu.en-si.si $SI_ROOT/OpenSubtitles2018.en-si.si > $DESTDIR/train.$si-$en.$si
cat $SI_ROOT/GNOMEKDEUbuntu.en-si.en $SI_ROOT/OpenSubtitles2018.en-si.en > $DESTDIR/train.$si-$en.$en

cat $NE_ROOT/bible_dup.en-ne.ne $NE_ROOT/bible.en-ne.ne $NE_ROOT/globalvoices.2018q4.ne-en.ne $NE_ROOT/GNOMEKDEUbuntu.en-ne.ne $NE_ROOT/nepali-penn-treebank.ne >  $DESTDIR/train.$ne-$en.$ne
cat $NE_ROOT/bible_dup.en-ne.en $NE_ROOT/bible.en-ne.en $NE_ROOT/globalvoices.2018q4.ne-en.en $NE_ROOT/GNOMEKDEUbuntu.en-ne.en $NE_ROOT/nepali-penn-treebank.en >  $DESTDIR/train.$ne-$en.$en


#Download the test sets
wget https://github.com/facebookresearch/flores/raw/master/data/wikipedia_en_ne_si_test_sets.tgz
tar -xvzf wikipedia_en_ne_si_test_sets.tgz

cp wikipedia_en_ne_si_test_sets/wikipedia.dev.ne-en.ne $DESTDIR/valid.$ne-$en.$ne
cp wikipedia_en_ne_si_test_sets/wikipedia.dev.ne-en.en $DESTDIR/valid.$ne-$en.$en

cp wikipedia_en_ne_si_test_sets/wikipedia.dev.si-en.si $DESTDIR/valid.$si-$en.$si
cp wikipedia_en_ne_si_test_sets/wikipedia.dev.si-en.en $DESTDIR/valid.$si-$en.$en

cp wikipedia_en_ne_si_test_sets/wikipedia.devtest.ne-en.ne $DESTDIR/devtest.$ne-$en.$ne
cp wikipedia_en_ne_si_test_sets/wikipedia.devtest.ne-en.en $DESTDIR/devtest.$ne-$en.$en

cp wikipedia_en_ne_si_test_sets/wikipedia.devtest.si-en.si $DESTDIR/devtest.$si-$en.$si
cp wikipedia_en_ne_si_test_sets/wikipedia.devtest.si-en.en $DESTDIR/devtest.$si-$en.$en

cp wikipedia_en_ne_si_test_sets/wikipedia.test.ne-en.ne $DESTDIR/test.$ne-$en.$ne
cp wikipedia_en_ne_si_test_sets/wikipedia.test.ne-en.en $DESTDIR/test.$ne-$en.$en

cp wikipedia_en_ne_si_test_sets/wikipedia.test.si-en.si $DESTDIR/test.$si-$en.$si
cp wikipedia_en_ne_si_test_sets/wikipedia.test.si-en.en $DESTDIR/test.$si-$en.$en

rm -rf wikipedia_en_ne_si_test_sets.tgz wikipedia_en_ne_si_test_sets
