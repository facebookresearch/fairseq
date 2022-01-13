#!/bin/bash

set -u

val_sets="dev_other"
graph_name=graph
decode_suffix=""
decode_script="steps/decode_fmllr.sh"
decode_args=""
nj=60

. ./cmd.sh
. ./path.sh
. parse_options.sh

set -x
exp_dir=$1
data_root=$2
lang_test=$3

graph=$exp_dir/$graph_name

if [ ! -d $graph ]; then
  utils/mkgraph.sh $lang_test $exp_dir $graph
fi

for part in $val_sets; do
  dec_dir=$exp_dir/decode${decode_suffix}_${part}
  if [ ! -d $dec_dir ]; then
    echo "decoding $part for $exp_dir"
    $decode_script --nj $nj --cmd "$decode_cmd" $decode_args \
      $graph $data_root/$part $dec_dir &
  else
    echo "$dec_dir exists. skip"
  fi
done

wait
