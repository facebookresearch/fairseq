#!/bin/bash
set -e
model=$1
mosesdecoder="/private/home/xutaima/tools/mosesdecoder"
fairseq_dir=$(dirname "$0")/../../..
infer_log=$1.infer_dev.log
data=~/data/ast/must_c_1_0/en-ro/data
CUDA_VISIBLE_DEVICES=1 python $fairseq_dir/examples/speech_recognition/infer.py\
	$data \
    --task speech_recognition \
    --max-tokens 25000 \
    --nbest 1 \
    --results-path $infer_log \
    --batch-size 512 \
    --user-dir $fairseq_dir/examples/speech_recognition \
	--path $model \
    --gen-subset valid\
    --beam 10 \
    --kspmodel $data/spm/spm_bpe_5000.model \

$mosesdecoder/scripts/generic/multi-bleu.perl $infer_log/ref.units-*-valid.txt < $infer_log/hypo.units-*-valid.txt | tee $infer_log/multi-bleu-units

