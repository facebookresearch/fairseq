#!/bin/bash
set -e
model=$3
mosesdecoder="/private/home/xutaima/tools/mosesdecoder"
fairseq_dir=$(dirname "$0")/../../..
infer_log=$3.infer_$1_$4.log
data=$2
CUDA_VISIBLE_DEVICES=1 python $fairseq_dir/examples/speech_recognition/infer.py\
	$data \
    --task speech_translation \
    --max-tokens 25000 \
    --nbest 1 \
    --results-path $infer_log \
    --batch-size 512 \
    --user-dir $fairseq_dir/examples/simultaneous_translation \
	--path $model \
    --gen-subset $1\
    --beam $4 \

sacrebleu $infer_log/ref.word*-$1.txt < $infer_log/hypo.word*-$1.txt | tee $infer_log/bleu-$1
wer $infer_log/ref.word*-$1.txt  $infer_log/hypo.word*-$1.txt | tee $infer_log/wer-$1
python $fairseq_dir/examples/simul_speech_translation/utils/eval_latency.py \
    --input ./$infer_log/hypo.delays-*txt

