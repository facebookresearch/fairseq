#!/bin/bash

# prepare a new data directory of HMM word output

. ./path.sh

set -eu

out_dir=  # same as in train.sh
dec_lmparam=  # LM hyperparameters (e.g., 7.0.0)

dec_exp=tri3b  # what HMM stage to decode (e.g., tri3b)
dec_suffix=word
dec_splits="train valid"
dec_data_dir=$out_dir/dec_data_word  # where to write HMM output

data_dir=$out_dir/data
wrd_data_dir=$out_dir/data_word

for x in $dec_splits; do
  mkdir -p $dec_data_dir/$x
  cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $dec_data_dir/$x/

  tra=$out_dir/exp/$dec_exp/decode${dec_suffix}_${x}/scoring/${dec_lmparam}.tra
  cat $tra | utils/int2sym.pl -f 2- $data_dir/lang_word/words.txt | \
    sed 's:<UNK>::g' | sed 's:<SIL>::g' > $dec_data_dir/$x/text
  utils/fix_data_dir.sh $dec_data_dir/$x
  echo "WER on $x is" $(compute-wer ark:$wrd_data_dir/${x}_gt/text ark:$dec_data_dir/$x/text | cut -d" " -f2-)
done

