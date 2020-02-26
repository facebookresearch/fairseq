#!/bin/bash
set -e
config=$1
. $1

model=$2

num_test=$(wc -l $src | cut -d " " -f1)
echo Config: $(realpath $config)
echo Model: $(realpath $model)
echo Source: $test_src
echo Target: $test_tgt
echo Tokenizer: $tokenizer

user_dir=$(dirname "$0")/..

port=$3

echo Server Port $port

echo Evaluatation starts at $(date +%Y/%m/%d-%H:%M:%S) &&
python $user_dir/eval/evaluate.py \
    --port $port \
    --agent-type simul_trans_text \
    --data-bin $data_bin \
    --src-splitter-type $src_splitter_type \
    --src-splitter-path $src_splitter_path \
    --tgt-splitter-type $tgt_splitter_type \
    --tgt-splitter-path $tgt_splitter_path \
    --model-path $model $extra_args \
    --reset-server \
    --scores
echo Evaluation ends at $(date +%Y/%m/%d-%H:%M:%S)



