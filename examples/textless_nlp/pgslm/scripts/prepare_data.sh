#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -eu

train_json=$1
valid_json=$2
test_json=$3
n_units=$4
hop_size=$5
sr=$6
f0_quantizer=$7
out_dir=$8

meta_path="$out_dir/data_config.json"
f0_dir="$out_dir/f0"

mkdir -p $out_dir
ln -sf $train_json $out_dir/train.txt
ln -sf $valid_json $out_dir/valid.txt
ln -sf $test_json $out_dir/test.txt

cat <<EOF >$meta_path
{
    "manifests": {
      "train": "$out_dir/train.txt",
      "valid": "$out_dir/valid.txt",
      "test": "$out_dir/test.txt"
    },
    "n_units": $n_units,
    "code_hop_size": $hop_size,
    "sampling_rate": $sr,
    "multispkr": "parent_parent_name",

    "f0_vq_type": "naive",
    "f0_vq_naive_quantizer": {
      "log_mean_norm": "$f0_quantizer"
    },
    "f0_vq_n_units": 32
}
EOF

for split in train valid test; do
  python examples/textless_nlp/pgslm/preprocess_f0.py \
    $out_dir/$split.txt $f0_dir/$split --nshards=1 --rank=1 --sampling_rate=$sr

  #NSHARDS=16
  #seq 1 $NSHARDS | parallel -j $NSHARDS python examples/textless_nlp/pgslm/preprocess_f0.py \
  #  $out_dir/$split.txt $f0_dir/$split --nshards=$NSHARDS --sampling_rate=$sr --rank
done

# Please make sure that the number of shards (--nshards_list) is consistent across commands
python examples/textless_nlp/pgslm/prepare_dataset.py \
  $meta_path $f0_dir --splits test valid train --nshards_list 1
