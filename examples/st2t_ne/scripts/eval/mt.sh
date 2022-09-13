#!/bin/bash

mt_prefix_folder=$1

for l in es fr it ; do
  echo " ------- Processing $l ------- "
  /private/home/mgaido/scripts/ne/eval/sacrebleu.sh /large_experiments/ust/mgaido/2022/data/s2t/en-$l/test.$l ${mt_prefix_folder}_$l/generate-test.txt
  /private/home/mgaido/scripts/ne/eval/ne_acc.sh ${mt_prefix_folder}_$l/generate-test.txt $l $l
done
