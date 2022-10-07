#!/bin/bash

input_folder=$1

for l in es fr it ; do
  python /private/home/mgaido/fairseq/fairseq_cli/generate.py $input_folder \
    --task multilingual_translation \
    --model-overrides '{"max_source_positions": 8196, "dataset_impl": "raw"}' --dataset-impl raw --max-source-positions 8196 \
    --max-tokens 20000 --nbest 1 --gen-subset test --lang-pairs en-es,en-fr,en-it \
    --source-lang en --target-lang $l \
    --sacrebleu --remove-bpe 'sentencepiece' \
    --beam 5 --path /checkpoint/mgaido/2022/ne/mt/fntags_multilang/avg10.pt --results-path $input_folder/mtneemb_test_$l
done
