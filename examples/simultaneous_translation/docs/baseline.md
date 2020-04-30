# **Baseline Simultaneous Translation**
---

This is an instruction of training and evaluating a *wait-k* simultanoes LSTM model on MUST-C English-Gernam Dataset.

[STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework](https://https://www.aclweb.org/anthology/P19-1289/)


## **Requirements**
Install fairseq (make sure to use the correct branch):
```
git clone --branch simulastsharedtask git@github.com:pytorch/fairseq.git
cd fairseq
pip install -e .
```

Assuming that fairseq is installed in a directory called `FAIRSEQ`.

Install SentencePiece. One easy way is to use anaconda:

```
conda install -c powerai sentencepiece
```

Download the MuST-C data for English-German available at https://ict.fbk.eu/must-c/.
We will assume that the data is downloaded in a directory called `DATA_ROOT`.


## **Text-to-text Model**
---
### Data Preparation
Train a SentencePiece model:
```shell
for lang in en de; do
    python $FAIRSEQ/examples/simultaneous_translation/data/train_spm.py \
        --data-path $DATA_ROOT/data \
        --vocab-size 10000 \
        --max-frame 3000 \
        --model-type unigram \
        --lang $lang \
        --out-path .
```

Process the data with the SentencePiece model:
```shell
proc_dir=proc
mkdir -p $proc_dir
for split in train dev tst-COMMON tst-HE; do
    for lang in en de; do
        spm_encode \
            --model unigram-$lang-10000-3000/spm.model \
            < $DATA_ROOT/data/$split/txt/$split.$lang \
            > $proc_dir/$split.spm.$lang
    done
done
```

Binarize the data:

```shell
proc_dir=proc
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $proc_dir/train.spm \
    --validpref $proc_dir/dev.spm \
    --testpref $proc_dir/tst-COMMON.spm \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --workers 20 \
    --destdir ./data-bin/mustc_en_de \
```

### Training


```shell
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=1 python $FAIRSEQ/train.py data-bin/mustc_en_de \
    --save-dir checkpoints \
    --arch berard_simul_text_iwslt \
    --simul-type waitk \
    --waitk-lagging 2 \
    --optimizer adam \
    --max-epoch 100 \
    --lr 0.001 \
    --clip-norm 5.0  \
    --max-sentences 128  \
    --log-format json \
    --log-interval 10 \
    --criterion cross_entropy_acc \
    --user-dir $FAIRSEQ/examples/simultaneous_translation
```

## **Speech-to-text Model**
---
### Data Preparation
First, segment wav files.
```shell 
python $FAIRSEQ/examples/simultaneous_translation/data/segment_wav.py \
    --datapath $DATA_ROOT
```
Similar to text-to-text model, train a Sentencepiecemodel, but only train on German
```Shell
python $FAIRSEQ/examples/simultaneous_translation/data/train_spm.py \
    --data-path $DATA_ROOT/data \
    --vocab-size 10000 \
    --max-frame 3000 \
    --model-type unigram \
    --lang $lang \
    --out-path .
```
## Training
```shell
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=1 python $FAIRSEQ/train.py data-bin/mustc_en_de \
    --save-dir checkpoints \
    --arch berard_simul_text_iwslt \
    --waitk-lagging 2 \
    --waitk-stride 10 \
    --input-feat-per-channel 40 \
    --encoder-hidden-size 512 \
    --output-layer-dim 128 \
    --decoder-num-layers 3 \
    --task speech_translation \
    --user-dir $FAIRSEQ/examples/simultaneous_translation
    --optimizer adam \
    --max-epoch 100 \
    --lr 0.001 \
    --clip-norm 5.0  \
    --max-sentences 128  \
    --log-format json \
    --log-interval 10 \
    --criterion cross_entropy_acc \
    --user-dir $FAIRSEQ/examples/simultaneous_translation
```

## Evaluation
---
### Evaluation Server
For text translation models, the server is set up as follow give input file and reference file. 

``` shell
python ./eval/server.py \
    --hostname localhost \
    --port 12321 \
    --src-file $DATA_ROOT/data/dev/txt/dev.en \
    --ref-file $DATA_ROOT/data/dev/txt/dev.de
```
For speech translation models, the input is the data direcrory.
``` shell
python ./eval/server.py \
    --hostname localhost \
    --port 12321 \
    --ref-file $DATA_ROOT \
    --data-type speech
```

### Decode and Evaluate with Client
Once the server is set up, run client to evaluate translation quality and latency.
```shell
# TEXT
python $fairseq_dir/examples/simultaneous_translation/evaluate.py \
    data-bin/mustc_en_de \
    --user-dir $FAIRSEQ/examples/simultaneous_translation \
    --src-spm unigram-en-10000-3000/spm.model\
    --tgt-spm unigram-de-10000-3000/spm.model\
    -s en -t de \
    --path checkpoints/checkpoint_best.pt

# SPEECH
python $fairseq_dir/examples/simultaneous_translation/evaluate.py \
    data-bin/mustc_en_de \
    --user-dir $FAIRSEQ/examples/simultaneous_translation \
    --data-type speech \
    --tgt-spm unigram-de-10000-3000/spm.model\
    -s en -t de \
    --path checkpoints/checkpoint_best.pt
```
