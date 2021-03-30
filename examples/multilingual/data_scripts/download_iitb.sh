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

IITB=$WORKDIR_ROOT/IITB
mkdir -p $IITB
pushd $IITB 

wget http://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/iitb_corpus_download/parallel.tgz
tar -xvzf parallel.tgz 

wget http://www.cfilt.iitb.ac.in/~moses/iitb_en_hi_parallel/iitb_corpus_download/dev_test.tgz
tar -xvzf dev_test.tgz 

DESTDIR=${WORKDIR_ROOT}/ML50/raw/
 
cp parallel/IITB.en-hi.en $DESTDIR/train.hi_IN-en_XX.en_XX
cp parallel/IITB.en-hi.hi $DESTDIR/train.hi_IN-en_XX.hi_IN

cp dev_test/dev.en $DESTDIR/valid.hi_IN-en_XX.en_XX
cp dev_test/dev.hi $DESTDIR/valid.hi_IN-en_XX.hi_IN

cp dev_test/test.en $DESTDIR/test.hi_IN-en_XX.en_XX
cp dev_test/test.hi $DESTDIR/test.hi_IN-en_XX.hi_IN
popd