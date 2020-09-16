#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Prepare librispeech dataset

set -x
set -e
fairseq_root=/private/home/yuntang/sourcecode/fairinternal2020H2/st_joint_train
base_url=www.openslr.org/resources/12
CURDIR=$PWD
OUTPUT="$CURDIR/outputs"
CKPTS="$CURDIR/ckpt"
LOGS=$CURDIR/logs
DIR_RAW_LIBRI=/datasets01/librispeech/062419
DIR_TO_SAVE_RAW_DATA=/checkpoint/yuntang/2020/T70609114_st_joint_train/fbank/
DIR_FOR_PREPROCESSED_DATA=$CURDIR/outputs
src_lang=en
nbpe=10000
bpemode=unigram
do_download_raw_data=0
do_feat_extraction=0


train_dir=train_960

parallel_num=50

mkdir -p $CKPTS

mkdir -p $DIR_FOR_PREPROCESSED_DATA
cd $DIR_FOR_PREPROCESSED_DATA || exit

tsv_dir=$DIR_FOR_PREPROCESSED_DATA/${bpemode}-${nbpe}
mkdir -p $tsv_dir


if [ ! -d "$fairseq_root" ]; then
    echo "$0: Please set correct fairseq_root"
    exit 1
fi

if [[ $do_download_raw_data -eq 1 ]]; then
    echo "Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        url=$base_url/$part.tar.gz
        if ! wget -P $DIR_RAW_LIBRI $url; then
            echo "$0: wget failed for $url"
            exit 1
        fi
        if ! tar -C $DIR_RAW_LIBRI -xvzf $DIR_RAW_LIBRI/$part.tar.gz; then
            echo "$0: error un-tarring archive $DIR_RAW_LIBRI/$part.tar.gz"
            exit 1
        fi
    done
fi

if [[ $do_feat_extraction -eq 1 ]]; then
    echo "extract fbank features and saved in zip files"
    mkdir -p ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/{${train_dir},test-clean,test-other,dev-clean,dev-other,valid}
    rm -f $DIR_TO_SAVE_RAW_DATA/LibriSpeech/*/data.tsv
    if [ -e $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${train_dir}/text ]; then
        rm -f $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${train_dir}/text
    fi

    # Training data
    for part in train-clean-100 train-clean-360 train-other-500; do
        python $fairseq_root/examples/$exampledir/data/extract_features.py --audio-format flac --audio-dirs $DIR_RAW_LIBRI/${part} --data-path $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${train_dir}/${part} --data-tsv $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${train_dir}/${part}.tsv --parallel-num $parallel_num  --parallel-logdir $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${train_dir}/logs/${part} 
        cat $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${train_dir}/${part}.tsv >> $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${train_dir}/data.tsv
    done
    echo "Merge train text"
    for part in train-clean-100 train-clean-360 train-other-500; do
        find $DIR_RAW_LIBRI/${part} -name '*.txt' -exec cat {} \; >> ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/${train_dir}/text
    done
    
    # Test and dev sets
    for part in dev-clean dev-other test-clean test-other; do
        python $fairseq_root/examples/$exampledir/data/extract_features.py --audio-format flac --audio-dirs $DIR_RAW_LIBRI/${part} --data-path $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${part}/data --data-tsv $DIR_TO_SAVE_RAW_DATA/LibriSpeech/${part}/data.tsv  
        find $DIR_RAW_LIBRI/${part}/ -name '*.txt' -exec cat {} \; > ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/${part}/text
    done
    # Use combined dev-clean and dev-other as validation set
    cat ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/dev-clean/text ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/dev-other/text > ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/valid/text
    cat ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/dev-clean/data.tsv ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/dev-other/data.tsv > ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/valid/data.tsv
fi

fairseq_dict=data/lang_char/${bpemode}-${src_lang}-${nbpe}/dict.txt
bpemodel=data/lang_char/${bpemode}-${src_lang}-${nbpe}/vocab/spm_${bpemode}_${nbpe}.model
echo "dictionary: $fairseq_dict"
echo "Dictionary preparation"
ckpt=$CKPTS/dict
if [ ! -e $ckpt ]; then
    mkdir -p data/lang_char/
    cut -f 2- -d" " ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/${train_dir}/text > data/lang_char/input.txt

    python $fairseq_root/examples/$exampledir/data/train_spm.py \
        --data-path data/lang_char/input.txt \
        --vocab-size $nbpe \
        --model-type $bpemode \
        --lang en \
        --out-path data/lang_char 

    cp ${fairseq_dict} $tsv_dir/tgt_dict.txt
    cp ${bpemodel} $tsv_dir/spm.model
    pushd $PWD ; cd $tsv_dir
    ln -s tgt_dict.txt dict.txt
    popd &> /dev/null 
    touch $ckpt
fi


echo "Prepare tsv"
ckpt=$CKPTS/gen_tsvs
if [ ! -e $ckpt ]; then
    for part in test-other test-clean dev-other dev-clean valid ${train_dir}; do
        python ${fairseq_root}/examples/$exampledir/data/asr_prep_tsv.py --audio-tsv ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/${part}/data.tsv --tgt-labels ${DIR_TO_SAVE_RAW_DATA}/LibriSpeech/${part}/text --tgt-spm ${tsv_dir}/spm.model --output-tsv $tsv_dir/${part}.tsv
    done
    touch $ckpt
fi

ckpt=$CKPTS/gcmvn
if [ ! -e $ckpt ]; then
    python ${fairseq_root}/examples/$exampledir/data/gcmn_estimate.py  --tsv-path $tsv_dir/${train_dir}.tsv --gcmvn-path $tsv_dir/gcmvn.pkl  
    touch $ckpt
fi


