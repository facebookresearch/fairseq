#!/bin/bash

# decode into phones (and prepare a new data directory for HMM outputs)

. ./path.sh

set -eu

out_dir=  # same as in train.sh
dec_lmparam=  # LM hyperparameters (e.g., 7.0.0)
dec_exp=
dec_script=
dec_splits="train valid"
dec_data_dir=$out_dir/dec_data  # where to write HMM output

data_dir=${out_dir}/data

local/decode.sh --nj 40 --graph_name graph \
  --val_sets "$dec_splits" --decode_script $dec_script \
  $out_dir/exp/$dec_exp $data_dir $data_dir/lang_test

if [ ! -z $dec_lmparam ]; then
  for x in $dec_splits; do
    mkdir -p $dec_data_dir/$x
    cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $dec_data_dir/$x/
  
    tra=$out_dir/exp/$dec_exp/decode_${x}/scoring/${dec_lmparam}.tra
    cat $tra | utils/int2sym.pl -f 2- $data_dir/lang/words.txt | \
      sed 's:<UNK>::g' | sed 's:<SIL>::g' > $dec_data_dir/${x}/text
    utils/fix_data_dir.sh $dec_data_dir/${x}
    echo "WER on ${x} is" $(compute-wer ark:$data_dir/${x}_gt/text ark:$dec_data_dir/$x/text | cut -d" " -f2-)
  done
fi
