#!/bin/bash

asr_folder=$1

#for l in es fr it ; do
for l in es ; do
  # for w in 0.05 0.1 0.15 0.2 ; do
  for w in 0.15 ; do
    python /private/home/mgaido/fairseq/fairseq_cli/generate.py /large_experiments/ust/mgaido/2022/data/joint_s2t/st/ \
      --task speech_text_joint_to_text --load-speech-only \
      --model-overrides '{"max_source_positions": 1000000}' --max-source-positions 1000000 \
      --max-tokens 1000000 --nbest 1 --batch-size 10 --gen-subset test_ep_ph_${l}_asr \
      --config-yaml config_asr.yaml --beam 5 --lenpen 1.0 --user-dir examples/speech_text_joint_to_text \
      --lm-is-class-based --lm-path /checkpoint/mgaido/2022/ne/lm/testonly/en/avg10_best.pt --lm-weight $w \
      --path $asr_folder/avg10.pt --results-path $asr_folder/test_ep_avg10_testonlylm_$w
  done
done
