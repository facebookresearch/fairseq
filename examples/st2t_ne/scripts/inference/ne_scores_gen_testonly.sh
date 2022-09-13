#!/bin/bash

folder_name=$1
st_or_asr=$2
l=$3

python /private/home/mgaido/fairseq/examples/speech_text_joint_to_text/scripts/entity_retrieval_scores.py /large_experiments/ust/mgaido/2022/data/joint_s2t/st \
  --task speech_text_joint_to_text --user-dir /private/home/mgaido/fairseq/examples/speech_text_joint_to_text \
  --noise-token ▁NOISE --max-tokens 3000000 --max-source-positions 3000000 \
  --config-yaml config_${st_or_asr}.yaml --gen-subset test_ep_ph_${l}_${st_or_asr} \
  --list-candidates /private/home/mgaido/models/improvin_speech_translation/en-es/list_test_entities.ph \
  --scores-activation-function sigmoid \
  --path /checkpoint/mgaido/2022/ne/$folder_name/avg5.pt --results-path /checkpoint/mgaido/2022/ne/$folder_name/scores_testonly_test_ep_${l}.pickle

python /private/home/mgaido/fairseq/examples/speech_text_joint_to_text/scripts/sym_scores_analysis.py \
  --tsv-ref /private/home/mgaido/datasets/ne/NEuRoparl-ST-v1.0/dataset/annotated-txt/en-${l}/test.ne.en.iob \
  --sym-scores /checkpoint/mgaido/2022/ne/$folder_name/scores_testonly_test_ep_${l}.pickle \
  --outputfig /checkpoint/mgaido/2022/ne/$folder_name/en${l}_testonly_recall.png \
  --list-candidates /private/home/mgaido/models/improvin_speech_translation/en-es/list_test_entities.txt \
  --lang en > /checkpoint/mgaido/2022/ne/$folder_name/recall_testonly_${l}.log 2>&1


