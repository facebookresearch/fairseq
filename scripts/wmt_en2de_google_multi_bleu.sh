#!/bin/bash

if [ $# -ne 1 ]; then
    echo "usage: $0 GENERATE_PY_OUTPUT"
    exit 1
fi
echo -e "\n RUN >> "$0

script_root=./scripts
detokenizer=$script_root/detokenizer.perl
replace_unicode_punctuation=$script_root/replace-unicode-punctuation.perl
tokenizer=$script_root/tokenizer.perl
multi_bleu=$script_root/multi-bleu.perl

GEN=$1
SYS=$GEN.sys
REF=$GEN.ref
if [ $(tail -n 1 $GEN | grep BLEU | wc -l) -ne 1 ]; then
	echo "not done generating"
	exit
fi

grep ^H $GEN | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $SYS
grep ^T $GEN | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $REF

#detokenize the decodes file to format the manner to do tokenize
perl $detokenizer -l de < $SYS > $SYS.dtk
perl $detokenizer -l de < $REF > $REF.dtk

#replace unicode
perl $replace_unicode_punctuation -l de < $SYS.dtk > $SYS.dtk.punc
perl $replace_unicode_punctuation -l de < $REF.dtk > $REF.dtk.punc

#tokenize the decodes file by moses tokenizer.perl
perl $tokenizer -l de < $SYS.dtk.punc > $SYS.dtk.punc.tok
perl $tokenizer -l de < $REF.dtk.punc > $REF.dtk.punc.tok

#"rich-text format" --> rich ##AT##-##AT## text format.
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $SYS.dtk.punc.tok > $SYS.dtk.punc.tok.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $REF.dtk.punc.tok > $REF.dtk.punc.tok.atat

perl $multi_bleu $REF.dtk.punc.tok.atat < $SYS.dtk.punc.tok.atat

rm -f $SYS.dtk $SYS.dtk.punc $SYS.dtk.punc.tok $REF.dtk $REF.dtk.punc $REF.dtk.punc.tok