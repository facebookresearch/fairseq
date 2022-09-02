#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -eu

train_json=$1
sr=$2
nbins=$3
out_dir=$4
out_prefix=$5

f0_dir="$out_dir/f0"

python examples/textless_nlp/pgslm/preprocess_f0.py \
    $train_json $f0_dir/${out_prefix}_f0_quant --nshards 1 --rank 1 --sampling_rate $sr

# NB: one can use parallel here:
# NSHARDS=16
#
#seq 1 $NSHARDS | parallel -j $NSHARDS python examples/textless_nlp/pgslm/preprocess_f0.py \
#    $train_json $f0_dir/${out_prefix}_f0_quant --nshards $NSHARDS --sampling_rate $sr --rank

python examples/textless_nlp/pgslm/quantize_f0.py \
    $train_json $f0_dir/${out_prefix}_f0_quant $out_dir $out_prefix --nbins $nbins --nshards 1 --normalize mean --log
