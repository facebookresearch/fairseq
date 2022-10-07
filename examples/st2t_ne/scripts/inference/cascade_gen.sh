#!/bin/bash

asr_folder=$1


for l in es fr it ; do
  /private/home/mgaido/scripts/ne/eval/extract_out.py <  $asr_folder/generate-test_ep_ph_${l}_asr.txt | sed -r 's:</?[A-Z_]+>::g' > $asr_folder/test_ep_ph_${l}_asr_notags.txt
  python /private/home/mgaido/scripts/spm_apply.py /large_experiments/ust/mgaido/2022/data/joint_s2t/spm_model_multilang.model \
    $asr_folder/test_ep_ph_${l}_asr_notags.txt $asr_folder/test_ep_${l}.en-$l.en

  cp /large_experiments/ust/mgaido/2022/data/mt/test.en-$l.$l $asr_folder/test_ep_${l}.en-$l.$l
  ln -s /large_experiments/ust/mgaido/2022/data/mt/dict* $asr_folder
  python /private/home/mgaido/fairseq/fairseq_cli/generate.py $asr_folder \
    --task multilingual_translation \
    --model-overrides '{"max_source_positions": 8196, "dataset_impl": "raw"}' --dataset-impl raw --max-source-positions 8196 \
    --max-tokens 20000 --nbest 1 --gen-subset test_ep_${l} --lang-pairs en-es,en-fr,en-it \
    --source-lang en --target-lang $l \
    --sacrebleu --remove-bpe 'sentencepiece' \
    --beam 5 --path /checkpoint/mgaido/2022/ne/mt/base_multilang/avg10.pt --results-path $asr_folder/cascade_base
done
