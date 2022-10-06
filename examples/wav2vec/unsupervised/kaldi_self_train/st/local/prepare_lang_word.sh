#!/bin/bash

num_sil_states=3
num_nonsil_states=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

set -eux

dict=$1
data_dir=$2
lexicon=$3

dict_dir=$data_dir/local/dict_word
tmplm_dir=$data_dir/local/lang_tmp_word
lm_dir=$data_dir/lang_word

mkdir -p $dict_dir $tmplm_dir $lm_dir

# prepare dict
echo "SIL" > $dict_dir/silence_phones.txt
echo "SIL" > $dict_dir/optional_silence.txt
awk '{print $1}' $dict > $dict_dir/nonsilence_phones.txt

(echo "!SIL SIL"; echo "<UNK> SIL";) | cat - $lexicon > $dict_dir/lexicon.txt

echo "SIL" > $dict_dir/extra_questions.txt
awk '{printf $1" "} END {printf "\n"}' $dict >> $dict_dir/extra_questions.txt

# prepare lang
utils/prepare_lang.sh --position-dependent-phones false \
  --num_sil_states $num_sil_states --num_nonsil_states $num_nonsil_states \
  $dict_dir "<UNK>" $tmplm_dir $lm_dir
