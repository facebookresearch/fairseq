#!/bin/bash
set -e
config=$1
. $1

model=$2

if [ $scorer_type = "text" ]
then
    num_test=$(wc -l $src | cut -d " " -f1)
else
    num_test=$(grep length_ms $tgt | wc -l)
fi
echo $num_test

chunk_size=300
num_chunk=$((num_test / chunk_size))
echo Config: $(realpath $config)
echo Model: $(realpath $model)
echo Source: $src
echo Target: $tgt
echo Tokenizer: $tokenizer
echo Agent: $agent_type

user_dir=$(dirname "$0")/..

echo Evaluatation starts at $(date +%Y/%m/%d-%H:%M:%S) &&
python $user_dir/eval/evaluate_local.py \
    --scorer-type $scorer_type \
    --agent-type $agent_type \
    --tokenizer $tokenizer \
    --src-file $src \
    --tgt-file $tgt \
    --data-bin $data_bin \
    --model-path $model \
    --src-splitter-type $src_splitter_type \
    --src-splitter-path $src_splitter_path \
    --tgt-splitter-type $tgt_splitter_type \
    --tgt-splitter-path $tgt_splitter_path \
    --num-threads 10
echo Evaluation ends at $(date +%Y/%m/%d-%H:%M:%S)



