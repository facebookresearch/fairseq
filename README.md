# fairseq-py
FAIR Sequence-to-Sequence (PyTorch)

## Setup

Build PyTorch from https://github.com/colesbury/pytorch/tree/fairseq-py:

```
git clone https://github.com/colesbury/pytorch.git -b fairseq-py
cd pytorch
python setup.py build develop
```

See instructions at https://github.com/pytorch/pytorch#from-source

## Training examples

To train IWSLT de-en:

```
DATADIR=/mnt/vol/gfsai-flash-east/ai-group/users/sgross/fairseq/iwslt14_de-en/
python train.py $DATADIR -a fconv_iwslt_de_en --lr 0.25 --clip-norm 0.1 --dropout 0.2
```

To train WMT'16 en-ro:

```
DATADIR=/mnt/vol/gfsai-flash-east/ai-group/users/sgross/fairseq/wmt16_en2ro/
python train.py $DATADIR -a fconv_wmt_en_ro --lr 0.25 --clip-norm 0.1 --dropout 0.1 -b 64 --max-len 1600
```
