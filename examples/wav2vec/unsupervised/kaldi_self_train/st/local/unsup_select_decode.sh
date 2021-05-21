#!/bin/bash

split="dev_other"
ref_txt=""  # ground truth transcript path
psd_txt=""  # pseudo transcript path
get_best_wer=true
dec_name="decode"
graph_name="graph"
kenlm_path=/checkpoint/abaevski/data/speech/libri/librispeech_lm_novox.phnc_o6.bin

. ./cmd.sh
. ./path.sh
. parse_options.sh

exp_root=$1
unsup_args=""
if [ $# -ge 2 ]; then
  unsup_args=$2
fi

set -eu

if [ ! -z $ref_txt ] && $get_best_wer; then
  echo "==== WER w.r.t. real transcript (select based on unsupervised metric)"
  for x in $exp_root/*/${dec_name}_${split}*; do
    lang=$(dirname $x)/$graph_name

    (
      for tra in $x/scoring/*.tra; do
        cat $tra | utils/int2sym.pl -f 2- $lang/words.txt | sed 's:<UNK>::g' | sed 's:<SIL>::g' > $tra.txt
        python local/unsup_select.py $psd_txt $tra.txt --kenlm_path $kenlm_path --gt_tra $ref_txt $unsup_args
      done 2>/dev/null | grep "score=" | sed 's/=/ /g' | sed 's/;//g' | sort -k3n | head -n1
    ) &
  done
fi
wait

