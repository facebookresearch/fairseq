#!/bin/bash

st_folder=$1


for l in es fr it ; do
  for w in 0.05 0.1 0.15 0.2 ; do
    python /private/home/mgaido/fairseq/fairseq_cli/generate.py /large_experiments/ust/mgaido/2022/data/joint_s2t/st/ \
      --task speech_text_joint_to_text --load-speech-only --infer-target-lang $l \
      --model-overrides '{"max_source_positions": 1000000}' --max-source-positions 1000000 \
      --max-tokens 1000000 --nbest 1 --batch-size 10 --gen-subset test_ep_ph_${l}_st \
      --config-yaml config_st.yaml --beam 5 --lenpen 1.0 --user-dir examples/speech_text_joint_to_text \
      --lm-is-class-based --lm-path /checkpoint/mgaido/2022/ne/lm/testonly/$l/avg10_best.pt:/checkpoint/mgaido/2022/ne/generic_lm/$l/fn/avg5_best.pt --lm-weight $w \
      --path $st_folder/avg10.pt --results-path $st_folder/test_ep_avg10_testonlylmplusgeneric_$w
  done
done
