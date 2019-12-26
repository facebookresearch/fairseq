# Text-to-text baseline

## Requirements

Install fairseq (make sure to use the correct branch):

```
git clone --branch simulastsharedtask git@github.com:pytorch/fairseq.git
cd fairseq
pip install -e .
```

We will assume that fairseq is installed in a directory called `FAIRSEQ`.

Install SentencePiece. One easy way is to use anaconda:

```
conda install -c powerai sentencepiece
```

## Download the data

Download the MuST-C data for English-German available at https://ict.fbk.eu/must-c/.
We will assume that the data is downloaded in a directory called `DATA_ROOT`.

## Data Preparation

Train a SentencePiece model:
```
for lang in en de; do
    python $FAIRSEQ/examples/simulast_shared_task/data/train_spm.py --data-path $DATA_ROOT/data --vocab-size 10000 --model-type unigram --lang $lang --out-path .
```

Process the data with the SentencePiece model:
```
proc_dir=proc
mkdir -p $proc_dir
for split in train dev tst-COMMON tst-HE; do
    for lang in en de; do
        spm_encode --model unigram-$lang-10000-3000/spm.model < $DATA_ROOT/data/$split/txt/$split.$lang > $proc_dir/$split.spm.$lang
    done
done
```

Binarize the data:

```
proc_dir=proc
fairseq-preprocess --source-lang en --target-lang de --trainpref $proc_dir/train.spm --validpref $proc_dir/dev.spm --testpref $proc_dir/tst-COMMON.spm --destdir ./data-bin/mustc_en_de --thresholdtgt 0 --thresholdsrc 0 --workers 20
```

## Training

```
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
