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


SRCDIR=$WORKDIR_ROOT/indic_languages_corpus
DESTDIR=${WORKDIR_ROOT}/ML50/raw/
mkdir -p $SRCDIR
mkdir -p $DESTDIR

cd $SRCDIR
wget http://lotus.kuee.kyoto-u.ac.jp/WAT/indic-multilingual/indic_languages_corpus.tar.gz
tar -xvzf indic_languages_corpus.tar.gz

SRC_EXTRACT_DIR=$SRCDIR/indic_languages_corpus/bilingual

cp $SRC_EXTRACT_DIR/ml-en/train.ml $DESTDIR/train.ml_IN-en_XX.ml_IN
cp $SRC_EXTRACT_DIR/ml-en/train.en $DESTDIR/train.ml_IN-en_XX.en_XX
cp $SRC_EXTRACT_DIR/ml-en/dev.ml $DESTDIR/valid.ml_IN-en_XX.ml_IN
cp $SRC_EXTRACT_DIR/ml-en/dev.en $DESTDIR/valid.ml_IN-en_XX.en_XX
cp $SRC_EXTRACT_DIR/ml-en/test.ml $DESTDIR/test.ml_IN-en_XX.ml_IN
cp $SRC_EXTRACT_DIR/ml-en/test.en $DESTDIR/test.ml_IN-en_XX.en_XX

cp $SRC_EXTRACT_DIR/ur-en/train.ur $DESTDIR/train.ur_PK-en_XX.ur_PK
cp $SRC_EXTRACT_DIR/ur-en/train.en $DESTDIR/train.ur_PK-en_XX.en_XX
cp $SRC_EXTRACT_DIR/ur-en/dev.ur $DESTDIR/valid.ur_PK-en_XX.ur_PK
cp $SRC_EXTRACT_DIR/ur-en/dev.en $DESTDIR/valid.ur_PK-en_XX.en_XX
cp $SRC_EXTRACT_DIR/ur-en/test.ur $DESTDIR/test.ur_PK-en_XX.ur_PK
cp $SRC_EXTRACT_DIR/ur-en/test.en $DESTDIR/test.ur_PK-en_XX.en_XX

cp $SRC_EXTRACT_DIR/te-en/train.te $DESTDIR/train.te_IN-en_XX.te_IN
cp $SRC_EXTRACT_DIR/te-en/train.en $DESTDIR/train.te_IN-en_XX.en_XX
cp $SRC_EXTRACT_DIR/te-en/dev.te $DESTDIR/valid.te_IN-en_XX.te_IN
cp $SRC_EXTRACT_DIR/te-en/dev.en $DESTDIR/valid.te_IN-en_XX.en_XX
cp $SRC_EXTRACT_DIR/te-en/test.te $DESTDIR/test.te_IN-en_XX.te_IN
cp $SRC_EXTRACT_DIR/te-en/test.en $DESTDIR/test.te_IN-en_XX.en_XX
