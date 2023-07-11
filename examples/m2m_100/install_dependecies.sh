#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


CWD=`pwd`
INSTALL_PATH=$CWD/tokenizers/thirdparty

MOSES=$INSTALL_PATH/mosesdecoder
if [ ! -d $MOSES ]; then
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git $MOSES
    cd $MOSES
    # To deal with differences in handling ' vs "
    git checkout 03578921cc1a03402
    cd -
fi

WMT16_SCRIPTS=$INSTALL_PATH/wmt16-scripts
if [ ! -d $WMT16_SCRIPTS ]; then
    echo 'Cloning Romanian tokenization scripts'
    git clone https://github.com/rsennrich/wmt16-scripts.git $WMT16_SCRIPTS
fi

KYTEA=$INSTALL_PATH/kytea
if [ ! -f $KYTEA/bin/kytea ]; then
    git clone https://github.com/neubig/kytea.git $KYTEA
    cd $KYTEA
    autoreconf -i
    ./configure --prefix=`pwd`
    make
    make install
    cd ..
fi

export MECAB=$INSTALL_PATH/mecab-0.996-ko-0.9.2
if [ ! -f $MECAB/bin/mecab ]; then
    cd $INSTALL_PATH
    curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
    tar zxfv mecab-0.996-ko-0.9.2.tar.gz
    cd mecab-0.996-ko-0.9.2/
    ./configure --prefix=`pwd`
    make
    make install

    cd ..
    curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
    tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz
    cd mecab-ko-dic-2.1.1-20180720/
    ./autogen.sh
    ./configure --prefix=`pwd` --with-dicdir=$MECAB/lib/mecab/dic/mecab-ko-dic --with-mecab-config=$MECAB/bin/mecab-config
    make
    sh -c 'echo "dicdir=$MECAB/lib/mecab/dic/mecab-ko-dic" > $MECAB/etc/mecabrc'
    make install
    cd $CWD
fi

INDIC_RESOURCES_PATH=$INSTALL_PATH/indic_nlp_resources
if [ ! -d $INDIC_RESOURCES_PATH ]; then
    echo 'Cloning indic_nlp_resources'
    git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git $INDIC_RESOURCES_PATH
fi


if [ ! -f $INSTALL_PATH/seg_my.py ]; then
    cd $INSTALL_PATH
    wget http://lotus.kuee.kyoto-u.ac.jp/WAT/my-en-data/wat2020.my-en.zip
    unzip wat2020.my-en.zip
    # switch to python3
    cat wat2020.my-en/myseg.py  |sed 's/^sys.std/###sys.std/g' | sed 's/### sys/sys/g' | sed 's/unichr/chr/g' > seg_my.py
    cd $CWD
fi


pip install pythainlp sacrebleu indic-nlp-library

