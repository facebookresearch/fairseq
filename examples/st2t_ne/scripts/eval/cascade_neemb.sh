#!/bin/bash

st_folder=$1

for l in es fr it ; do
  echo "Processing $l"
  /private/home/mgaido/scripts/ne/eval/sacrebleu.sh /large_experiments/ust/mgaido/2022/data/s2t/en-$l/test.$l $st_folder/generate-test_ep_neemb_${l}.txt
  /private/home/mgaido/scripts/ne/eval/ne_acc.sh $st_folder/generate-test_ep_neemb_${l}.txt $l $l
done
