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

If you want to skip training and evaluate pre-trained models, you can go to the Evaluation section.


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
./scripts/start-client.sh \
    ./scripts/configs/must-c-en_de-speech-text.sh \
    ./experiments/checkpoints/checkpoint_best.pt
```
To evaluate speech translation models on dev set. 
```shell
./scripts/start-client.sh \
    ./scripts/configs/must-c-en_de-speech-dev.sh \
    ./experiments/checkpoints/checkpoint_best.pt
```

We also provide a faster evaluation script that splits the dataset and launches multiple clients. For example for speech translation,
```shell
./scripts/start-multi-client.sh \
    ./scripts/configs/must-c-en_de-speech-dev.sh \
    ./experiments/checkpoints/checkpoint_best.pt
```

### Pretrained models

You can use the client scripts with pre-trained models:
* [text model](https://dl.fbaipublicfiles.com/simultaneous_translation/checkpoint_text_waitk3.pt). You should obtain the following score: `Scores: {"BLEU": 13.291, "TER": 0.957, "METEOR": 0.31, "DAL": 4.372044027815046, "AL": 2.5295724889866804, "AP": 0.6400225334686246}`

| k  | BLEU | TER | METEOR | DAL | AL | AP |
| -- | ---- | --- | ------ | --- | -- | -- |
| 1  | 7.326 | 1.058 | 0.19 | 2.168934043745549 | -0.991472002729109 | 0.4658645800748763 |
| 2  | 12.154 | 0.97 | 0.277 | 3.2513923756856777 | 1.1303054744050929 | 0.5664211640203829 |
| 3  | 13.291 | 0.957 | 0.31 | 4.372044027815046 | 2.5295724889866804 | 0.6400225334686246 |
| 4  | 14.169 | 0.99 | 0.319 | 4.861679422880509 | 2.910830587016724 | 0.6664540746307507 |
| 5  | 15.145 | 0.815 | 0.332 | 5.304197753388866 | 3.121181273391799 | 0.6845044787614512 |
| 6  | 14.803 | 0.961 | 0.328 | 6.188106422263472 | 4.214606605907574 | 0.7318075373611732 |
| 7  | 15.994 | 0.963 | 0.344 | 7.0365747842500515 | 5.300415757428318 | 0.7697175975184749 |
| 8  | 17.275 | 0.874 | 0.354 | 7.519929816667579 | 5.877349037344574 | 0.7907821372287507 |
| 9  | 17.034 | 0.924 | 0.356 | 8.170097741122323 | 6.52711183232735 | 0.8136009025536033 |
| 10 | 17.639 | 0.952 | 0.359 | 8.77114769361929 | 7.157067218550644 | 0.832459437914156 |
| 20 | 19.53 | 0.837 | 0.388 | 13.549994174477423 | 12.372683254204748 | 0.9382112652000539 |
| 1000 | 23.037 | 0.728 | 0.435 | 19.85172198845816 | 19.85172171468728 | 1.0 |


* [speech model](https://dl.fbaipublicfiles.com/simultaneous_translation/checkpoint_speech_waitk_lag5_stride10.pt). You should obtain the following scores: `{"BLEU": 10.785, "TER": 0.913, "METEOR": 0.247, "DAL": 2817.45349595572, "AL": 2331.9959397710254, "AP": 0.8462297623865175}`
