#!/bin/bash
set -e
model=$3
mosesdecoder="/private/home/xutaima/tools/mosesdecoder"
fairseq_dir=$(dirname "$0")/../../..
infer_log=$3.infer_$1_$4.log
data=$2
CUDA_VISIBLE_DEVICES=1 python $fairseq_dir/examples/speech_recognition/infer.py\
	$data \
    --task speech_recognition \
    --max-tokens 25000 \
    --nbest 1 \
    --results-path $infer_log \
    --batch-size 512 \
    --user-dir $fairseq_dir/examples/speech_recognition \
	--path $model \
    --gen-subset $1\
    --beam $4 \
    --criterion ctc_loss \
    --w2l-decoder ctc_greedy

wer $infer_log/ref.word*-$1.txt $infer_log/hypo.word*-$1.txt | tee $infer_log/wer-$1

