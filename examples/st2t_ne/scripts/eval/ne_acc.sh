#!/bin/bash

gen_txt=$1
testlang=$2
targetlang=$3


/private/home/mgaido/scripts/ne/eval/extract_out.py < $gen_txt | sed -r 's:</?[A-Z_]+>::g' | sed 's/#CONTEXT#//g' > $gen_txt.raw

python /private/home/mgaido/scripts/ne/ne_terms_accuracy.py --input $gen_txt.raw \
  --tsv-ref /private/home/mgaido/datasets/ne/NEuRoparl-ST-v1.0/dataset/annotated-txt/en-$testlang/test.ne.$targetlang.iob \
  --lang $targetlang | grep -e '^Overall' -e '^GPE' -e '^LOC' -e '^PERSON' -e '^ORG' | tail -n 5

