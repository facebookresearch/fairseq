#!/bin/bash

tmp_dir=$(mktemp -d -t ci-XXXXXXXXXX)
ref=$1
gen_out=$2

/private/home/mgaido/scripts/ne/eval/extract_out.py < $gen_out > $gen_out.raw
sed -r 's:</?[A-Z_]+>::g' $gen_out.raw | /private/home/mgaido/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | /private/home/mgaido/mosesdecoder/scripts/tokenizer/lowercase.perl -l en | tr -d '[:punct:]' > $tmp_dir/out.txt
sed -r 's:</?[A-Z_]+>::g' $ref | /private/home/mgaido/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | /private/home/mgaido/mosesdecoder/scripts/tokenizer/lowercase.perl -l en | tr -d '[:punct:]' > $tmp_dir/ref.txt

/private/home/mgaido/scripts/levenshtein -in1 $tmp_dir/ref.txt -in2 $tmp_dir/out.txt > $tmp_dir/wer.score
tail -1 $tmp_dir/wer.score
