#!/bin/bash
set -e
config=$1
. $1

model=$2

num_test=$(wc -l $src | cut -d " " -f1)
echo Config: $(realpath $config)
echo Model: $(realpath $model)
echo Source: $src
echo Target: $tgt
echo Tokenizer: $tokenizer
echo Agent: $agent_type

user_dir=$(dirname "$0")/..

if [ -z "$3" ]
  then
    port=12321
else
    port=$3
fi

echo Port $port

echo Server Port $port

echo Evaluatation starts at $(date +%Y/%m/%d-%H:%M:%S) &&
python $user_dir/eval/evaluate.py \
    --port $port \
    --agent-type $agent_type \
    --data-bin $data_bin \
    --src-splitter-type $src_splitter_type \
    --src-splitter-path $src_splitter_path \
    --tgt-splitter-type $tgt_splitter_type \
    --tgt-splitter-path $tgt_splitter_path \
    --model-path $model $extra_args \
    --reset-server \
    --num-threads 1 \
    --scores
echo Evaluation ends at $(date +%Y/%m/%d-%H:%M:%S)



