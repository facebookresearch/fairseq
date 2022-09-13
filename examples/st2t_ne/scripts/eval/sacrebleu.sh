#!/bin/bash

tmp_dir=$(mktemp -d -t ci-XXXXXXXXXX)
ref=$1
gen_out=$2

/private/home/mgaido/scripts/ne/eval/extract_out.py < $gen_out > $gen_out.raw
sed -r 's:</?[A-Z_]+>::g' $gen_out.raw | sed 's/#CONTEXT#//g' > $tmp_dir/out.txt
sed -r 's:</?[A-Z_]+>::g' $ref > $tmp_dir/ref.txt


sacrebleu $tmp_dir/ref.txt -i $tmp_dir/out.txt


