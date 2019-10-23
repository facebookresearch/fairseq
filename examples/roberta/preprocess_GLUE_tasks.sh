#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# raw glue data as downloaded by glue download script (https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
if [[ $# -ne 2 ]]; then
  echo "Run as following:"
  echo "./examples/roberta/preprocess_GLUE_tasks.sh <glue_data_folder> <task_name>"
  exit 1
fi

GLUE_DATA_FOLDER=$1
TASKS=$2 # QQP

# download bpe encoder.json, vocabulary and fairseq dictionary
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

if [ "$TASKS" = "ALL" ]
then
  TASKS="QQP MNLI QNLI MRPC RTE STS-B SST-2 CoLA BoolQ"
fi

for TASK in $TASKS
do
  echo "Preprocessing $TASK"

  TASK_DATA_FOLDER="$GLUE_DATA_FOLDER/$TASK"
  echo "Raw data as downloaded from glue website: $TASK_DATA_FOLDER"

  SPLITS="train dev test"
  INPUT_COUNT=2
  if [ "$TASK" = "QQP" ]
  then
    INPUT_COLUMNS=( 4 5 )
    TEST_INPUT_COLUMNS=( 2 3 )
    LABEL_COLUMN=6
  elif [ "$TASK" = "MNLI" ]
  then
    SPLITS="train dev_matched dev_mismatched test_matched test_mismatched"
    INPUT_COLUMNS=( 9 10 )
    TEST_INPUT_COLUMNS=( 9 10 )
    DEV_LABEL_COLUMN=16
    LABEL_COLUMN=12
  elif [ "$TASK" = "QNLI" ]
  then
    INPUT_COLUMNS=( 2 3 )
    TEST_INPUT_COLUMNS=( 2 3 )
    LABEL_COLUMN=4
  elif [ "$TASK" = "MRPC" ]
  then
    INPUT_COLUMNS=( 4 5 )
    TEST_INPUT_COLUMNS=( 4 5 )
    LABEL_COLUMN=1
  elif [ "$TASK" = "RTE" ]
  then
    INPUT_COLUMNS=( 2 3 )
    TEST_INPUT_COLUMNS=( 2 3 )
    LABEL_COLUMN=4
  elif [ "$TASK" = "STS-B" ]
  then
    INPUT_COLUMNS=( 8 9 )
    TEST_INPUT_COLUMNS=( 8 9 )
    LABEL_COLUMN=10
  elif [ "$TASK" = "BoolQ" ]
  then
    INPUT_COLUMNS=( 1 2 )
    TEST_INPUT_COLUMNS=( 1 2 )
    LABEL_COLUMN=3
  # Following are single sentence tasks.
  elif [ "$TASK" = "SST-2" ]
  then
    INPUT_COLUMNS=( 1 )
    TEST_INPUT_COLUMNS=( 2 )
    LABEL_COLUMN=2
    INPUT_COUNT=1
  elif [ "$TASK" = "CoLA" ]
  then
    INPUT_COLUMNS=( 4 )
    TEST_INPUT_COLUMNS=( 2 )
    LABEL_COLUMN=2
    INPUT_COUNT=1
  fi

  # Strip out header and filter lines that don't have expected number of fields.
  rm -rf "$TASK_DATA_FOLDER/processed"
  mkdir -p "$TASK_DATA_FOLDER/processed"
  for SPLIT in $SPLITS
  do
    # CoLA train and dev doesn't have header.
    if [[ ( "$TASK" = "CoLA") && ( "$SPLIT" != "test" ) ]]
    then
      cp "$TASK_DATA_FOLDER/$SPLIT.tsv" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
    else
      tail -n +2 "$TASK_DATA_FOLDER/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
    fi

    # Remove unformatted lines from train and dev files for QQP dataset.
    if [[ ( "$TASK" = "QQP") && ( "$SPLIT" != "test" ) ]]
    then
      awk -F '\t' -v NUM_FIELDS=6 'NF==NUM_FIELDS{print}{}' "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp" > "$TASK_DATA_FOLDER/processed/$SPLIT.tsv";
    else
      cp "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv";
    fi
    rm "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
  done

  # Split into input0, input1 and label
  for SPLIT in $SPLITS
  do
    for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
    do
      if [[ "$SPLIT" != test* ]]
      then
        COLUMN_NUMBER=${INPUT_COLUMNS[$INPUT_TYPE]}
      else
        COLUMN_NUMBER=${TEST_INPUT_COLUMNS[$INPUT_TYPE]}
      fi
      cut -f"$COLUMN_NUMBER" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.raw.input$INPUT_TYPE";
    done

    if [[ "$SPLIT" != test* ]]
    then
      if [ "$TASK" = "MNLI" ] && [ "$SPLIT" != "train" ]
      then
        cut -f"$DEV_LABEL_COLUMN" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv"  > "$TASK_DATA_FOLDER/processed/$SPLIT.label";
      else
        cut -f"$LABEL_COLUMN" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.label";
      fi
    fi

    # BPE encode.
    for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
    do
      LANG="input$INPUT_TYPE"
      echo "BPE encoding $SPLIT/$LANG"
      python -m examples.roberta.multiprocessing_bpe_encoder \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$TASK_DATA_FOLDER/processed/$SPLIT.raw.$LANG" \
      --outputs "$TASK_DATA_FOLDER/processed/$SPLIT.$LANG" \
      --workers 60 \
      --keep-empty;
    done
  done

  # Remove output directory.
  rm -rf "$TASK-bin"

  DEVPREF="$TASK_DATA_FOLDER/processed/dev.LANG"
  TESTPREF="$TASK_DATA_FOLDER/processed/test.LANG"
  if [ "$TASK" = "MNLI" ]
  then
    DEVPREF="$TASK_DATA_FOLDER/processed/dev_matched.LANG,$TASK_DATA_FOLDER/processed/dev_mismatched.LANG"
    TESTPREF="$TASK_DATA_FOLDER/processed/test_matched.LANG,$TASK_DATA_FOLDER/processed/test_mismatched.LANG"
  fi

  # Run fairseq preprocessing:
  for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
  do
    LANG="input$INPUT_TYPE"
    fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.$LANG" \
      --validpref "${DEVPREF//LANG/$LANG}" \
      --testpref "${TESTPREF//LANG/$LANG}" \
      --destdir "$TASK-bin/$LANG" \
      --workers 60 \
      --srcdict dict.txt;
  done
  if [[ "$TASK" !=  "STS-B" ]]
  then
    fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.label" \
      --validpref "${DEVPREF//LANG/label}" \
      --destdir "$TASK-bin/label" \
      --workers 60;
  else
    # For STS-B output range is converted to be between: [0.0, 1.0]
    mkdir -p "$TASK-bin/label"
    awk '{print $1 / 5.0 }' "$TASK_DATA_FOLDER/processed/train.label" > "$TASK-bin/label/train.label"
    awk '{print $1 / 5.0 }' "$TASK_DATA_FOLDER/processed/dev.label" > "$TASK-bin/label/valid.label"
  fi
done

rm -rf 'encoder.json'
rm -rf 'vocab.bpe'
rm -rf 'dict.txt'
