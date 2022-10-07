#!/bin/bash

asr_out_folder=$1

for l in es fr it ; do
  echo " ------ Evaluation of ${l} ------"
  /private/home/mgaido/scripts/ne/eval/wer.sh /large_experiments/ust/mgaido/2022/data/s2t/en-$l/test.en $asr_out_folder/generate-test_ep_ph_${l}_asr.txt
  /private/home/mgaido/scripts/ne/eval/ne_acc.sh $asr_out_folder/generate-test_ep_ph_${l}_asr.txt $l en
done

