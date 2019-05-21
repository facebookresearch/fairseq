# Learning Deep Transformer Models for Machine Translation on Fairseq

The implementation of [Learning Deep Transformer Models for Machine Translation [ACL 2019] ](todo) (Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, Lidia S. Chao)

> This code is based on [Fairseq v0.5.0](https://github.com/pytorch/fairseq/tree/v0.5.0)

## Installation

1. `pip install -r requirements.txt`
2. `python setup.py install`

## Prepare Training Data

1. Download the preprocessed [WMT'16 En-De dataset](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) provided by Google to project root dir

2. Generate binary dataset at `data-bin/wmt16_en_de_google`

> `bash runs/prepare-wmt-en2de.sh`

## Train

### Train deep pre-norm baseline (20-layer encoder)

> `bash runs/train-wmt-de2en-deep-prenorm-baseline.sh`

### Train deep post-norm DLCL (25-layer encoder)

> `bash runs/train-wmt-en2de-deep-postnorm-dlcl.sh`

### Train deep pre-norm DLCL (30-layer encoder)

> `bash runs/train-wmt-en2de-deep-prenorm-dlcl.sh`

NOTE: BLEU will be calculated automatically when finishing training

## Results

todo

## Acknowledgement

We thank Lei Dai for his great test works.