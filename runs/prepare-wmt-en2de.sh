#!/usr/bin/env bash
set -e

if [ ! -f wmt16_en_de.tar.gz ]; then
	echo "please download wmt16_en_de.tar.gz to project root dir"
	exit
fi

echo -e "| extract files ... \n"
data_dir=examples/translation/wmt_en_de
mkdir -p  $data_dir && tar zxvf wmt16_en_de.tar.gz -C $data_dir

ln -srf $data_dir/train.tok.clean.bpe.32000.en $data_dir/train.en
ln -srf $data_dir/train.tok.clean.bpe.32000.de $data_dir/train.de
ln -srf $data_dir/newstest2013.tok.bpe.32000.en $data_dir/valid.en
ln -srf $data_dir/newstest2013.tok.bpe.32000.de $data_dir/valid.de
ln -srf $data_dir/newstest2014.tok.bpe.32000.en $data_dir/test.en
ln -srf $data_dir/newstest2014.tok.bpe.32000.de $data_dir/test.de

echo -e "| preprocess ... it will take some time \n"
python preprocess.py \
--source-lang en --target-lang de \
--trainpref $data_dir/train \
--validpref $data_dir/valid \
--testpref $data_dir/test \
 --destdir data-bin/wmt16_en_de_google \
 --joined-dictionary \
 --thresholdsrc 5 --thresholdtgt 5