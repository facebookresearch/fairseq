#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# data should be downloaded and processed with reprocess_RACE.py
if [[ $# -ne 2 ]]; then
  echo "Run as following:"
  echo "./examples/roberta/preprocess_RACE.sh <race_data_folder> <output_folder>"
  exit 1
fi

RACE_DATA_FOLDER=$1
OUT_DATA_FOLDER=$2

# download bpe encoder.json, vocabulary and fairseq dictionary
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

SPLITS="train dev test-middle test-high"
INPUT_TYPES="input0 input1 input2 input3 input4"
for INPUT_TYPE in $INPUT_TYPES
do
  for SPLIT in $SPLITS
      do
      echo "BPE encoding $SPLIT/$INPUT_TYPE"
      python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json encoder.json \
            --vocab-bpe vocab.bpe \
            --inputs "$RACE_DATA_FOLDER/$SPLIT.$INPUT_TYPE" \
            --outputs "$RACE_DATA_FOLDER/$SPLIT.$INPUT_TYPE.bpe" \
            --workers 10 \
            --keep-empty;

      done
done

for INPUT_TYPE in $INPUT_TYPES
    do
      LANG="input$INPUT_TYPE"
      fairseq-preprocess \
        --only-source \
        --trainpref "$RACE_DATA_FOLDER/train.$INPUT_TYPE.bpe" \
        --validpref "$RACE_DATA_FOLDER/dev.$INPUT_TYPE.bpe" \
        --testpref "$RACE_DATA_FOLDER/test-middle.$INPUT_TYPE.bpe,$RACE_DATA_FOLDER/test-high.$INPUT_TYPE.bpe" \
        --destdir "$OUT_DATA_FOLDER/$INPUT_TYPE" \
        --workers 10 \
        --srcdict dict.txt;
done

rm -rf "$OUT_DATA_FOLDER/label"
mkdir -p "$OUT_DATA_FOLDER/label"
cp "$RACE_DATA_FOLDER/train.label" "$OUT_DATA_FOLDER/label/"
cp "$RACE_DATA_FOLDER/dev.label" "$OUT_DATA_FOLDER/label/valid.label"
cp "$RACE_DATA_FOLDER/test-middle.label" "$OUT_DATA_FOLDER/label/test.label"
cp "$RACE_DATA_FOLDER/test-high.label" "$OUT_DATA_FOLDER/label/test1.label"
