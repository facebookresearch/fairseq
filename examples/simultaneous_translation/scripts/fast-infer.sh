#!/bin/bash
set -e
mosesdecoder="/private/home/xutaima/tools/mosesdecoder"
config=$1
. $1

model=$2
output=$model.$split.results
mkdir -p $output

echo Config: $(realpath $config)
echo Model: $(realpath $model)
echo Source: $src
echo Target: $tgt
echo Tokenizer: $tokenizer

user_dir=.
batch_size=256

cat $src | CUDA_VISIBLE_DEVICES=1 python $user_dir/eval/fast_eval.py \
    $data_bin \
    --user-dir $user_dir \
    --path $model \
    --buffer-size $batch_size \
    --batch-size $batch_size --beam 1 \
    --remove-bpe \
    --result $output/pred
#$mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < $output/pred.text > $output/pred.detok.text
sacrebleu --tokenize $tokenizer $tgt < $output/pred.detok.text | tee $output/bleu
python $user_dir/eval/eval_latency.py --input $output/pred.delay | tee  $output/latency
cat $output/bleu $output/latency > $output/scores
