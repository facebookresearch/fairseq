#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

lg=$1
text_path=$2
target_dir=$3

ph_lg=${lg:l}
if test "$lg" = 'fr'; then
  ph_lg='fr-fr'
elif test "$lg" = 'en'; then
  ph_lg='en-us'
elif test "$lg" = 'pt'; then
  ph_lg='pt-br'
fi

echo $lg
echo $ph_lg
echo $text_path
echo $target_dir

mkdir -p $target_dir
python normalize_and_filter_text.py --lang $lg < $text_path | grep -v '\-\-\-' >! $target_dir/lm.upper.lid.txt
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/lm.upper.lid.txt --only-source --destdir $target_dir --thresholdsrc 2 --padding-factor 1 --dict-only
cut -f1 -d' ' $target_dir/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' >! $target_dir/words.txt

one=$(echo "1" | PHONEMIZER_ESPEAK_PATH=$(which espeak) phonemize -p ' ' -w '' -l $ph_lg --language-switch remove-flags)
sed 's/$/ 1/' $target_dir/words.txt | PHONEMIZER_ESPEAK_PATH=$(which espeak) phonemize -o $target_dir/phones.txt -p ' ' -w '' -l $ph_lg -j 70 --language-switch remove-flags

echo "one is ${one}"

sed -i "s/${one}$//" $target_dir/phones.txt
paste $target_dir/words.txt $target_dir/phones.txt >! $target_dir/lexicon.lst

python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones.txt --only-source --destdir $target_dir/phones --thresholdsrc 1000 --padding-factor 1 --dict-only

python filter_lexicon.py -d $target_dir/phones/dict.txt < $target_dir/lexicon.lst >! $target_dir/lexicon_filtered.lst
python phonemize_with_sil.py -s 0.25 --surround --lexicon $target_dir/lexicon_filtered.lst < $target_dir/lm.upper.lid.txt >! $target_dir/phones/lm.phones.filtered.txt
cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones/lm.phones.filtered.txt --workers 70 --only-source --destdir $target_dir/phones --srcdict $target_dir/phones/dict.phn.txt

lmplz -o 4 < $target_dir/lm.upper.lid.txt --discount_fallback --prune 0 0 0 3 >! $target_dir/kenlm.wrd.o40003.arpa
build_binary $target_dir/kenlm.wrd.o40003.arpa $target_dir/kenlm.wrd.o40003.bin
lg=$lg python examples/speech_recognition/kaldi/kaldi_initializer.py fst_dir=$target_dir/fst/phn_to_words_sil lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones "blank_symbol='<SIL>'"
lg=$lg python examples/speech_recognition/kaldi/kaldi_initializer.py fst_dir=$target_dir/fst/phn_to_words lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones

lmplz -o 4 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback >! $target_dir/phones/lm.phones.filtered.04.arpa
build_binary $target_dir/phones/lm.phones.filtered.04.arpa $target_dir/phones/lm.phones.filtered.04.bin
lmplz -o 6 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback >! $target_dir/phones/lm.phones.filtered.06.arpa
build_binary $target_dir/phones/lm.phones.filtered.06.arpa $target_dir/phones/lm.phones.filtered.06.bin

lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py fst_dir=$target_dir/fst/phn_to_phn_sil lm_arpa=$target_dir/phones/lm.phones.filtered.06.arpa data_dir=$target_dir/phones "blank_symbol='<SIL>'"
