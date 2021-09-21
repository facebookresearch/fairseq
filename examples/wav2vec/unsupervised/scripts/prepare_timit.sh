#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

timit_root=$1  # assume it is the upper-cased version
tgt_dir=$2
model=$3

set -eu

setups="matched unmatched"
splits="test valid train train_text"

tgt_dir=$(realpath $tgt_dir)
sph2wav=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
wav_dir=$tgt_dir/wav


mkdir -p $tgt_dir $wav_dir
find $timit_root/{TRAIN,TEST} -iname "*.WAV" > $tgt_dir/all_sph.flist
cat $tgt_dir/all_sph.flist | sed -e 's#//*#/#g' -e 's#.*/\([^/]*\)/\([^/]*\).WAV#\1_\2#g' > $tgt_dir/all.uid
paste -d' ' $tgt_dir/{all_sph.flist,all.uid} | \
  awk -v sph2wav=$sph2wav -v wav_dir=$wav_dir '{print sph2wav " -f wav " $1 " > " wav_dir "/" $2 ".wav"}' \
  > $tgt_dir/sph2wav.sh
bash $tgt_dir/sph2wav.sh
cat $tgt_dir/all.uid | awk -v wav_dir=$(pwd)/$wav_dir '{print $1" "wav_dir"/"$1".wav"}' | sort > $tgt_dir/all_wav.scp
cut -d' ' -f2 $tgt_dir/all_wav.scp | xargs -I{} soxi -s {} > $tgt_dir/all.dur
paste -d' ' $tgt_dir/{all_wav.scp,all.dur} > $tgt_dir/all_wav_dur.scp
rm $tgt_dir/{all.uid,all_sph.flist,sph2wav.sh}

find $timit_root/{TRAIN,TEST} -iname "*.PHN" > $tgt_dir/all_phn60.flist
while read line; do
  if [ ! -f $line ]; then 
    >&2 echo "Cannot find transcription file '$line'" && exit 1;
  fi
  cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
done < $tgt_dir/all_phn60.flist > $tgt_dir/all.phn60
cat $tgt_dir/all_phn60.flist | sed -e 's#//*#/#g' -e 's#.*/\([^/]*\)/\([^/]*\).PHN#\1_\2#g' | \
  paste -d' ' - $tgt_dir/all.phn60 | \
  $KALDI_ROOT/egs/timit/s5/local/timit_norm_trans.pl -i - -m $KALDI_ROOT/egs/timit/s5/conf/phones.60-48-39.map -to 39 | \
  sort > $tgt_dir/all.phn
echo "done preparing wav and 39-phone transcripts"


for s in $setups; do
  mkdir -p $tgt_dir/$s
  for x in $splits; do
    uid_path=config/timit_${s}/${x}.uid
    grep -w -f $uid_path $tgt_dir/all.phn | cut -d' ' -f2- > $tgt_dir/$s/$x.phn
    ln -sf $(realpath $tgt_dir/$s/$x.phn) $tgt_dir/$s/$x.wrd
    
    echo "/" > $tgt_dir/$s/$x.tsv &&  grep -w -f $uid_path $tgt_dir/all_wav_dur.scp | cut -d' ' -f2- | sed 's# #\t#'  >> $tgt_dir/$s/$x.tsv
  done
  
  for x in $splits; do
    cat $tgt_dir/$s/$x.phn
  done | tr ' ' '\n' | sort -u | awk '{print $1" "1}' > $tgt_dir/$s/dict.phn.txt
  ln -sf $(realpath $tgt_dir/$s/dict.phn.txt) $tgt_dir/$s/dict.wrd.txt
done
echo "done preparing unmatched and matched setups for TIMIT"


for s in $setups; do
  zsh scripts/prepare_audio.sh $tgt_dir/$s $tgt_dir/$s/feat $model

  lm_dir=$tgt_dir/$s/phones
  fst_dir=$tgt_dir/$s/fst/phn_to_phn

  python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $tgt_dir/$s/train_text.phn --workers 10 --only-source --destdir $lm_dir --srcdict $tgt_dir/$s/dict.phn.txt
  $KENLM_ROOT/lmplz -o 3 < $tgt_dir/$s/train_text.phn --discount_fallback >$lm_dir/train_text_phn.03.arpa
  $KENLM_ROOT/build_binary $lm_dir/train_text_phn.03.arpa $lm_dir/train_text_phn.03.bin
  $KENLM_ROOT/lmplz -o 4 < $tgt_dir/$s/train_text.phn --discount_fallback >$lm_dir/train_text_phn.04.arpa
  $KENLM_ROOT/build_binary $lm_dir/train_text_phn.04.arpa $lm_dir/train_text_phn.04.bin
  
  python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$fst_dir lm_arpa=$lm_dir/train_text_phn.03.arpa data_dir=$tgt_dir/$s in_labels=phn
done
echo "done preprocessing audio and text for wav2vec-U"
