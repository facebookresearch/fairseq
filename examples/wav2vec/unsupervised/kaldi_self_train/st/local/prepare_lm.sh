#!/usr/bin/env bash

langdir=""
lmdir=""

. ./cmd.sh
. ./path.sh
. parse_options.sh

arpa_lm=$1
data=$2

if [ -z $langdir ]; then
  langdir=$data/lang
fi
if [ -z $lmdir ]; then
  lmdir=$data/lang_test
fi

if [ ! -d $langdir ]; then
  echo "$langdir not found. run local/prepare_lang.sh first" && exit 1
fi

mkdir -p $lmdir
cp -r $langdir/* $lmdir

if [[ "$arpa_lm" == *.gz ]]; then
  gunzip -c $arpa_lm | arpa2fst --disambig-symbol=#0 --read-symbol-table=$lmdir/words.txt - $lmdir/G.fst
else
  arpa2fst --disambig-symbol=#0 --read-symbol-table=$lmdir/words.txt $arpa_lm $lmdir/G.fst
fi
fstisstochastic $lmdir/G.fst
utils/validate_lang.pl $lmdir || exit 1

echo "done preparing lm ($lmdir)"
