# Trivia Questions Tutorial based on original T5 notebook

The original T5 repository comes with an introduction notebook on finetuning previously pretrained T5 model.
This tutorial reproduces those results in Fairseq.

### Geting data
The dataset consists of Natural Questions and Trivia-QA datasets. The following script will download these datasets to `raw` directory. 

```bash
python download-data.py
```

### Data Preprocessing
The next step is to encode data using a [SentencePiece](https://github.com/google/sentencepiece) model. You can download the model from [here](https://storage.cloud.google.com/t5-data/vocabs/cc_all.32000/sentencepiece.model) or use the `gsutil`:

```bash
gsutil -m cp gs://t5-data/vocabs/cc_all.32000/sentencepiece.model
```

Moreover, we concatenate both train-sets into one.

```bash
mkdir -p encoded
cat raw/train-nq.questions raw/train-triviaqa.questions | spm_encode --model sentencepiece.model > encoded/train.questions
cat raw/train-nq.answers raw/train-triviaqa.answers | spm_encode --model sentencepiece.model > encoded/train.answers

cat raw/validation-nq.answers | spm_encode --model sentencepiece.model > encoded/validation-nq.answers
cat raw/validation-nq.questions | spm_encode --model sentencepiece.model > encoded/validation-nq.questions

cat raw/validation-triviaqa.answers | spm_encode --model sentencepiece.model > encoded/validation-triviaqa.answers
cat raw/validation-triviaqa.questions | spm_encode --model sentencepiece.model > encoded/validation-triviaqa.questions
```

### Data Binarization
The last step is data binazation.

```bash
fairseq-preprocess \
    --task t5-finetuning \
    --trainpref encoded/train \
    --validpref encoded/validation-nq,encoded/validation-triviaqa \
    --joined-dictionary \
    --destdir binarized \
    --dataset-impl mmap \
    --srcdict  sentencepiece.vocab \
    -s questions \
    -t answers
```

### Training
You can use any pretrained T5 model but the example below shows how to train with `t5-small`. First, download the model:

```bash
wget wget https://applica-public.s3-eu-west-1.amazonaws.com/fairseq-t5/t5-small.pt
```

Then we can start training:
```bash
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    --task t5-finetuning \
    --arch t5-small \
    --criterion cross_entropy \
    --z-loss 0.0001 \
    -s question \
    -t answer \
    --restore-file t5-small.pt \
    --max-sentences 256 \
    --max-source-positions 128 \
    --max-target-positions 32 \
    --vocab-path sentencepiece.model \
    --truncate-source \
    --lr 0.003 \
    --clip-norm 1 \
    --optimizer adafactor \
    --update-freq 7 \
    --loss-denominator 233472 \
    ./binarized
```
After a few epochs, the model should achieve about 12% ACC.
