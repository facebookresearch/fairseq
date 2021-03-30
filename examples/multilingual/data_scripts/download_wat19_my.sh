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
DESTDIR=$WORKDIR_ROOT/ML50/raw
mkdir -p $SRCDIR
mkdir -p $DESTDIR

WAT_MY_EN=wat2020.my-en.zip
cd $SRCDIR
# please refer to http://lotus.kuee.kyoto-u.ac.jp/WAT/my-en-data/ for latest URL if the following url expired
#- The data used for WAT2020 are identical to those used in WAT2019.
wget http://lotus.kuee.kyoto-u.ac.jp/WAT/my-en-data/$WAT_MY_EN
unzip $WAT_MY_EN


SRC_EXTRACT_DIR=$SRCDIR/wat2020.my-en/alt

cp $SRC_EXTRACT_DIR/train.alt.en $DESTDIR/train.my_MM-en_XX.en_XX
cp $SRC_EXTRACT_DIR/train.alt.my $DESTDIR/train.my_MM-en_XX.my_MM
cp $SRC_EXTRACT_DIR/dev.alt.en $DESTDIR/valid.my_MM-en_XX.en_XX
cp $SRC_EXTRACT_DIR/dev.alt.my $DESTDIR/valid.my_MM-en_XX.my_MM
cp $SRC_EXTRACT_DIR/test.alt.en $DESTDIR/test.my_MM-en_XX.en_XX
cp $SRC_EXTRACT_DIR/test.alt.my $DESTDIR/test.my_MM-en_XX.my_MM
