#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# set -x -e

if [ -z $WORKDIR_ROOT ] ;
then
        echo "please specify your working directory root in environment variable WORKDIR_ROOT. Exitting..."
        exit
fi

 
# put intermediate files
TMP_DIR=$WORKDIR_ROOT/temp/af_xhv2
# output {train,valid,test} files to dest
DEST=${WORKDIR_ROOT}/ML50/raw



ROOT=${WORKDIR_ROOT}
UTILS=$PWD/utils
TMX2CORPUS="${UTILS}/tmx2corpus"
TMX_TOOL="python ${TMX2CORPUS}/tmx2corpus.py"

mkdir -p $TMP_DIR
mkdir -p $DEST
mkdir -p $UTILS

function download_opus(){
    src=$1
    tgt=$2
    subset=$3
    ulr=$4

    mkdir extract_$subset.$src-$tgt
    pushd extract_$subset.$src-$tgt
    if [ ! -f "$subset.$src-$tgt.tmx.gz" ]; then
        wget $url -O "$subset.$src-$tgt.tmx.gz"
        gzip -d "$subset.$src-$tgt.tmx.gz"
        f=$subset.$src-$tgt.tmx
        $TMX_TOOL $f
        mv bitext.$src ../$subset.$src-$tgt.$src
        mv bitext.$tgt ../$subset.$src-$tgt.$tgt
    fi
    popd    
}

function concat_subsets(){
    src=$1
    tgt=$2
    subsets=$3
    src_train=raw_train.$src-$tgt.$src
    tgt_train=raw_train.$src-$tgt.$tgt
    > $src_train
    > $tgt_train
    for subset in $subsets; do
        cat $subset.$src-$tgt.$src >> $src_train
        cat $subset.$src-$tgt.$tgt >> $tgt_train
    done
}



function get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

function split_train_valid(){
    src=$1
    tgt=$2
    raw_src_train=raw_train.$src-$tgt.$src
    raw_tgt_train=raw_train.$src-$tgt.$tgt

    shuf --random-source=<(get_seeded_random 43) $raw_src_train > shuffled.$src-$tgt.$src 
    shuf --random-source=<(get_seeded_random 43) $raw_tgt_train > shuffled.$src-$tgt.$tgt 

    head -n 1500 shuffled.$src-$tgt.$src  > valid.$src-$tgt.$src
    head -n 1500 shuffled.$src-$tgt.$tgt > valid.$src-$tgt.$tgt

    tail +1501 shuffled.$src-$tgt.$src > train.$src-$tgt.$src
    tail +1501 shuffled.$src-$tgt.$tgt > train.$src-$tgt.$tgt     
}

function copy2dst(){
    lsrc=$1
    ltgt=$2
    src=${lsrc:0:2}
    tgt=${ltgt:0:2}
 

    cp valid.$src-$tgt.$src $DEST/valid.$lsrc-$ltgt.$lsrc 
    cp valid.$src-$tgt.$tgt $DEST/valid.$lsrc-$ltgt.$ltgt 

    cp train.$src-$tgt.$src $DEST/train.$lsrc-$ltgt.$lsrc 
    cp train.$src-$tgt.$tgt $DEST/train.$lsrc-$ltgt.$ltgt        
}




#for xh-en
declare -A xh_en_urls
xh_en_urls=(
    [Tatoeba]=https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/tmx/en-xh.tmx.gz 
    [wikimedia]=https://object.pouta.csc.fi/OPUS-wikimedia/v20190628/tmx/en-xh.tmx.gz
    [memat]=https://object.pouta.csc.fi/OPUS-memat/v1/tmx/en-xh.tmx.gz
    [uedin]=https://object.pouta.csc.fi/OPUS-bible-uedin/v1/tmx/en-xh.tmx.gz
    [GNOME]=https://object.pouta.csc.fi/OPUS-GNOME/v1/tmx/en-xh.tmx.gz
    [XhosaNavy]=https://object.pouta.csc.fi/OPUS-XhosaNavy/v1/tmx/en-xh.tmx.gz
    [KDE4]=https://object.pouta.csc.fi/OPUS-KDE4/v2/tmx/en-xh.tmx.gz
    [Ubuntu]=https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/tmx/en-xh.tmx.gz    
)

mkdir $TMP_DIR/xh-en
pushd $TMP_DIR/xh-en
for k in "${!xh_en_urls[@]}"
do
    name=$k
    url=${xh_en_urls[$k]}
    echo "$name: $url"
    download_opus xh en $name $ulr
done
concat_subsets xh en "${!xh_en_urls[@]}"
split_train_valid xh en
copy2dst xh_ZA en_XX
popd


##
#for af-en
declare -A af_en_urls
af_en_urls=(
    [Tatoeba]=https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/tmx/af-en.tmx.gz
    [uedin]=https://object.pouta.csc.fi/OPUS-bible-uedin/v1/tmx/af-en.tmx.gz
    [GNOME]=https://object.pouta.csc.fi/OPUS-GNOME/v1/tmx/af-en.tmx.gz
    [QED]=https://object.pouta.csc.fi/OPUS-QED/v2.0a/tmx/af-en.tmx.gz
    [KDE4]=https://object.pouta.csc.fi/OPUS-KDE4/v2/tmx/af-en.tmx.gz
    [OpenSubtitles]=https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/tmx/af-en.tmx.gz
    [SPC]=https://object.pouta.csc.fi/OPUS-SPC/v1/tmx/af-en.tmx.gz
    [Ubuntu]=https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/tmx/af-en.tmx.gz
)

mkdir $TMP_DIR/af-en
pushd $TMP_DIR/af-en
for k in "${!af_en_urls[@]}"
do
    name=$k
    url=${af_en_urls[$k]}
    echo "$name: $url"
    download_opus af en $name $ulr
done
concat_subsets af en "${!af_en_urls[@]}"
split_train_valid af en
copy2dst af_ZA en_XX
popd


