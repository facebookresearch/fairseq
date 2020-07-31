# Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)

This page includes instructions for reproducing results from the paper [Unsupervised Quality Estimation for Neural
Machine Translation (Fomicheva et al., 2020)](https://arxiv.org/abs/2005.10608)

## Requirements:

* mosesdecoder: https://github.com/moses-smt/mosesdecoder
* subword-nmt: https://github.com/rsennrich/subword-nmt
* flores: https://github.com/facebookresearch/flores

## Download Models and Test Data

Download translation models and test data from [MLQE dataset repository](https://github.com/facebookresearch/mlqe).

## Set up:

Given a testset consisting of source sentences and reference translations:

* `SRC_LANG`: source language
* `TGT_LANG`: target language
* `INPUT`: input prefix, such that the file `$INPUT.$SRC_LANG` contains source sentences and `$INPUT.$TGT_LANG`
contains the reference sentences
* `OUTPUT_DIR`: output path to store results
* `MOSES_DECODER`: path to mosesdecoder installation
* `BPE_ROOT`: path to subword-nmt installation
* `BPE`: path to BPE model
* `MODEL_DIR`: directory containing the NMT model `.pt` file as well as the source and target vocabularies.
* `TMP`: directory for intermediate temporary files
* `GPU`: if translating with GPU, id of the GPU to use for inference
* `DROPOUT_N`: number of stochastic forward passes

`$DROPOUT_N` is set to 30 in the experiments reported in the paper. However, we observed that increasing it beyond 10
does not bring substantial improvements.

## Translate the data using standard decoding

Preprocess the input data:
```
for LANG in $SRC_LANG $TGT_LANG; do
  perl $MOSES_DECODER/scripts/tokenizer/tokenizer.perl -threads 80 -a -l $LANG < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
  python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
done
```

Binarize the data for faster translation:

```
fairseq-preprocess --srcdict $MODEL_DIR/dict.$SRC_LANG.txt --tgtdict $MODEL_DIR/dict.$TGT_LANG.txt
--source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref $TMP/preprocessed.tok.bpe --destdir $TMP/bin --workers 4
```

Translate

```
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate $TMP/bin --path ${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}.pt --beam 5
--source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 > $TMP/fairseq.out
grep ^H $TMP/fairseq.out | cut -f3- > $TMP/mt.out
```

Post-process

```
sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/mt.out | perl $MOSES_DECODER/scripts/tokenizer/detokenizer.perl
-l $TGT_LANG > $OUTPUT_DIR/mt.out
```

## Produce uncertainty estimates

### Scoring

Make temporary files to store the translations repeated N times.

```
python ${SCRIPTS}/scripts/uncertainty/repeat_lines.py -i $TMP/preprocessed.tok.bpe.$SRC_LANG -n $DROPOUT_N
-o $TMP/repeated.$SRC_LANG
python ${SCRIPTS}/scripts/uncertainty/repeat_lines.py -i $TMP/mt.out -n $DROPOUT_N -o $TMP/repeated.$TGT_LANG

fairseq-preprocess --srcdict ${MODEL_DIR}/dict.${SRC_LANG}.txt $TGT_DIC --source-lang ${SRC_LANG}
--target-lang ${TGT_LANG} --testpref ${TMP}/repeated --destdir ${TMP}/bin-repeated
```

Produce model scores for the generated translations using `--retain-dropout` option to apply dropout at inference time:

```
CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate ${TMP}/bin-repeated --path ${MODEL_DIR}/${LP}.pt --beam 5
 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 --score-reference --retain-dropout
 --retain-dropout-modules TransformerModel TransformerEncoder TransformerDecoder TransformerEncoderLayer
 TransformerDecoderLayer --seed 46 > $TMP/dropout.scoring.out

grep ^H $TMP/dropout.scoring.out | cut -f2- > $TMP/dropout.scores

```

Use `--retain-dropout-modules` to specify the modules. By default, dropout is applied in the same places
as for training.

Compute the mean of the resulting output distribution:

```
python $SCRIPTS/scripts/uncertainty/aggregate_scores.py -i $TMP/dropout.scores -o $OUTPUT_DIR/dropout.scores.mean
-n $DROPOUT_N
```

### Generation

Produce multiple translation hypotheses for the same source using `--retain-dropout` option:

```
CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate ${TMP}/bin-repeated --path ${MODEL_DIR}/${LP}.pt
 --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --retain-dropout
 --unkpen 5 --retain-dropout-modules TransformerModel TransformerEncoder TransformerDecoder
TransformerEncoderLayer TransformerDecoderLayer --seed 46 > $TMP/dropout.generation.out

grep ^H $TMP/dropout.generation.out | cut -f3- > $TMP/dropout.hypotheses_

sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/dropout.hypotheses_ | perl $MOSES_DECODER/scripts/tokenizer/detokenizer.perl
-l $TGT_LANG > $TMP/dropout.hypotheses
```

Compute similarity between multiple hypotheses corresponding to the same source sentence using Meteor
evaluation metric:
```
python meteor.py -i $TMP/dropout.hypotheses -m <path_to_meteor_installation> -n $DROPOUT_N -o
$OUTPUT_DIR/dropout.gen.sim.meteor
```
