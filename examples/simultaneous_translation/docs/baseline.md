# **Baseline Simultaneous Translation**
---

Here are the instructions for training and evaluating a *wait-k* simultaneous LSTM model on the MUST-C English-German dataset.

[STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework](https://https://www.aclweb.org/anthology/P19-1289/)


## **Requirements**
Install fairseq (make sure to use the correct branch):
```
git clone --branch simulastsharedtask git@github.com:pytorch/fairseq.git
cd fairseq
pip install -e .
```

Assuming that fairseq is installed in a directory called `FAIRSEQ`. We will use `$FAIRSEQ/example/simultaneous_translation` as the work directory

Install SentencePiece. One easy way is to use anaconda:

```
conda install -c powerai sentencepiece
```

Download the MuST-C data for English-German available at https://ict.fbk.eu/must-c/.
In this example we will assume that the data is downloaded in a directory called ./experiments/data/must_c_1_0/en-de.


## **Text-to-text Model**
---
### Data Preparation
Train a SentencePiece model:
```shell
cd $FAIRSEQ/example/simultaneous_translation
DATA_ROOT=./experiments/data/must_c_1_0/en-de
for lang in en de; do
    python $FAIRSEQ/examples/simultaneous_translation/data/train_spm.py \
        --data-path $DATA_ROOT/data \
        --vocab-size 10000 \
        --max-frame 3000 \
        --model-type unigram \
        --lang $lang \
        --out-path $DATA_ROOT
```

Process the data with the SentencePiece model:
```shell
mkdir -p $DATA_ROOT/bi-text
for split in train dev tst-COMMON tst-HE; do
    for lang in en de; do
        spm_encode \
            --model $DATA_ROOT/unigram-$lang-10000-3000/spm.model \
            < $DATA_ROOT/data/$split/txt/$split.$lang \
            > $DATA_ROOT/bi-text/$split.spm.$lang
    done
done
```

Binarize the data:

```shell
proc_dir=$DATA_ROOT/bi-text
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $proc_dir/train.spm \
    --validpref $proc_dir/dev.spm \
    --testpref $proc_dir/tst-COMMON.spm \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --workers 20 \
    --destdir $DATA_ROOT/data-bin/mustc_en_de \
```

### Training


```shell
mkdir -p ./experiments/checkpoints
CUDA_VISIBLE_DEVICES=1 python $FAIRSEQ/train.py data-bin/mustc_en_de \
    --save-dir ./experiments/checkpoints \
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
cd $FAIRSEQ/example/simultaneous_translation
DATA_ROOT=./experiments/data/must_c_1_0/en-de
python $FAIRSEQ/examples/simultaneous_translation/data/segment_wav.py \
    --datapath $DATA_ROOT/data
```
Similar to text-to-text model, train a Sentencepiecemodel, but only train on German
```Shell
python $FAIRSEQ/examples/simultaneous_translation/data/train_spm.py \
    --data-path $DATA_ROOT/data \
    --vocab-size 10000 \
    --max-frame 3000 \
    --model-type unigram \
    --lang $lang \
    --out-path $DATA_ROOT
```
## Training
```shell
mkdir -p ./experiments/checkpoints
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
The server can evaluate different types of data given different configuration files
To evaluate text translation models on dev set. 

``` shell
./scripts/start-server.sh ./scripts/configs/must-c-en_de-text-dev.sh
```
To evaluate speech translation models on dev set.
``` shell
./scripts/start-server.sh ./scripts/configs/must-c-en_de-speech-dev.sh
```

### Decode and Evaluate with Client
Same as the server, one can use different configuration files to start different agent.
To evaluate text translation models on dev set. 
```shell
./script/start-client.py \
    ./scripts/configs/must-c-en_de-speech-text.sh \
    ./experiments/checkpoints/checkpoint_best.pt
```
To evaluate speech translation models on dev set. 
```shell
./script/start-client.py \
    ./scripts/configs/must-c-en_de-speech-dev.sh \
    ./experiments/checkpoints/checkpoint_best.pt
```

We also provide a faster evaluation script that splits the dataset and launches multiple clients. For example for speech translation,
```shell
./script/start-multi-client.py \
    ./scripts/configs/must-c-en_de-speech-dev.sh \
    ./experiments/checkpoints/checkpoint_best.pt
```
