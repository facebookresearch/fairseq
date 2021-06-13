#!/bin/bash

set -eu

w2v_dir=  # contains features `{train,valid}.{npy,lengths}`, real transcripts `{train,valid}.${label}`, and dict `dict.${label}.txt`
lab_dir=  # contains pseudo labels `{train,valid}.txt`
out_dir=  # output root
arpa_lm=  # phone LM
arpa_lm_bin=  # (binary) phone LM for KenLM, used in unsupervised selection

label=phnc
train_name="train"
valid_name="valid"
data_dir=${out_dir}/data

mkdir -p ${out_dir}/exp
local/prepare_lang.sh $w2v_dir/dict.${label}.txt $data_dir
local/prepare_lm.sh $arpa_lm $data_dir

for x in $train_name $valid_name; do
  x_gt=${x}_gt

  # prepare pseudo data
  python local/prepare_data_from_w2v.py $w2v_dir $data_dir $x
  steps/compute_cmvn_stats.sh $data_dir/$x $out_dir/exp/make_feat/$x $out_dir/feats/$x
  python local/copy_aligned_text.py < $lab_dir/$x.txt > $data_dir/$x/text

  # prepare ground truth data
  mkdir $data_dir/$x_gt
  cp $data_dir/$x/{feats.scp,cmvn.scp,utt2spk,spk2utt} $data_dir/$x_gt/
  python local/copy_aligned_text.py < $w2v_dir/$x.$label > $data_dir/$x_gt/text
done

local/train_subset_lgbeam.sh \
  --out_root ${out_dir} --out_name exp --train $train_name --valid $valid_name \
  --mono_size 2000 --tri1_size 5000 --tri2b_size -1 --tri3b_size -1 \
  --stage 1 --max_stage 3 $data_dir $data_dir/lang $data_dir/lang_test

local/unsup_select_decode.sh \
  --split $valid_name --kenlm_path $arpa_lm_bin \
  --ref_txt $data_dir/${valid_name}_gt/text \
  --psd_txt $data_dir/${valid_name}/text \
  $out_dir/exp
