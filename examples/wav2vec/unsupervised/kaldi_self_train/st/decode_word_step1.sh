#!/bin/bash

# prepare word WFSTs, reference data, and decode

set -eu

w2v_dir=  # same as in train.sh
out_dir=  # same as in train.sh
lexicon=  # word to phone mapping
wrd_arpa_lm=  # word LM
wrd_arpa_lm_bin=  # word LM for KenLM, used in unsupervised selection

dec_exp=  # what HMM stage to decode (e.g., tri3b)
dec_script=  # what decoding script to use (e.g., steps/decode_fmllr.sh)
phn_label=phnc
wrd_label=wrd
dec_suffix=word
dec_splits="train valid"
valid_split="valid"

data_dir=$out_dir/data
wrd_data_dir=$out_dir/data_word

lexicon_clean=$(mktemp)
cat $lexicon | sort | uniq > $lexicon_clean
local/prepare_lang_word.sh $w2v_dir/dict.${phn_label}.txt $data_dir $lexicon_clean && rm $lexicon_clean
local/prepare_lm.sh --langdir $data_dir/lang_word --lmdir $data_dir/lang_test_word $wrd_arpa_lm $data_dir

for x in $dec_splits; do
  x_gt=${x}_gt
  mkdir -p $wrd_data_dir/$x_gt
  cp $data_dir/$x_gt/{feats.scp,cmvn.scp,utt2spk,spk2utt} $wrd_data_dir/$x_gt/
  python local/copy_aligned_text.py < $w2v_dir/$x.$wrd_label > $wrd_data_dir/$x_gt/text
done

local/decode.sh --nj 40 --graph_name graph${dec_suffix} --decode_suffix $dec_suffix \
  --val_sets "$dec_splits" --decode_script $dec_script \
  $out_dir/exp/$dec_exp $data_dir $data_dir/lang_test_word

local/unsup_select_decode_word.sh \
  --split $valid_split --kenlm_path $wrd_arpa_lm_bin \
  --ref_txt $wrd_data_dir/${valid_split}_gt/text \
  --psd_txt $data_dir/${valid_split}/text \
  --dec_name decode${dec_suffix} --graph_name graph${dec_suffix} \
  --phonemize_lexicon $data_dir/local/dict_word/lexicon.txt \
  $out_dir/exp
