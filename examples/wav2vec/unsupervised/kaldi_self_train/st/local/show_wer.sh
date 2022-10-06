#!/bin/bash

split="dev_other"
ref_data=""
get_best_wer=true
dec_name="decode"
graph_name="graph"

. ./cmd.sh
. ./path.sh
. parse_options.sh

exp_root=$1

set -eu

echo "==== WER w.r.t. pseudo transcript"
for x in $exp_root/*/${dec_name}_${split}*; do grep WER $x/wer_* 2>/dev/null | utils/best_wer.sh; done


if [ ! -z $ref_data ]; then
  echo "==== WER w.r.t. real transcript (select based on pseudo WER)"
  ref_txt=$ref_data/$split/text
  for x in $exp_root/*/${dec_name}_${split}*; do
    lang=$(dirname $x)/$graph_name

    lmwt=$(
      grep WER $x/wer_* 2>/dev/null | utils/best_wer.sh |
      sed 's/.*wer_\(.*\)$/\1/g' | sed 's/_/./g'
    )
    tra=$x/scoring/$lmwt.tra
    cat $tra | utils/int2sym.pl -f 2- $lang/words.txt | sed 's:<UNK>::g' | sed 's:<SIL>::g' | \
      compute-wer --text --mode=present \
      ark:$ref_txt  ark,p:- 2> /dev/null | grep WER | xargs -I{} echo {} $tra
  done
fi

if [ ! -z $ref_data ] && $get_best_wer; then
  echo "==== WER w.r.t. real transcript (select based on true WER)"
  ref_txt=$ref_data/$split/text
  for x in $exp_root/*/${dec_name}_${split}*; do
    lang=$(dirname $x)/$graph_name

    for tra in $x/scoring/*.tra; do
      cat $tra | utils/int2sym.pl -f 2- $lang/words.txt | sed 's:<UNK>::g' | sed 's:<SIL>::g' | \
        compute-wer --text --mode=present \
        ark:$ref_txt  ark,p:- 2> /dev/null | grep WER | xargs -I{} echo {} $tra
    done | sort -k2n | head -n1
  done
fi

exit 0;
