# Neural Machine Translation

## Pre-trained models

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-de.newstest2014.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.v2.en-de.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381); WMT'18 winner) | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.bz2) | See NOTE in the archive

## Example usage

Generation with the binarized test sets can be run in batch mode as follows, e.g. for WMT 2014 English-French on a GTX-1080ti:
```
$ mkdir -p data-bin
$ curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ fairseq-generate data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
| Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Compute BLEU score
$ grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
$ fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

## Preprocessing

These scripts provide an example of pre-processing data for the NMT task.

### prepare-iwslt14.sh

Provides an example of pre-processing for IWSLT'14 German to English translation task: ["Report on the 11th IWSLT evaluation campaign" by Cettolo et al.](http://workshop2014.iwslt.org/downloads/proceeding.pdf)

Example usage:
```
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/iwslt14.tokenized.de-en
$ fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en

# Train the model (better for a single GPU setup):
$ mkdir -p checkpoints/fconv
$ CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 200 \
  --arch fconv_iwslt_de_en --save-dir checkpoints/fconv

# Generate:
$ fairseq-generate data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/fconv/checkpoint_best.pt \
  --batch-size 128 --beam 5 --remove-bpe

```

To train transformer model on IWSLT'14 German to English:
```
# Preparation steps are the same as for fconv model.

# Train the model (better for a single GPU setup):
$ mkdir -p checkpoints/transformer
$ CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
  -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s de -t en \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/transformer

# Average 10 latest checkpoints:
$ python scripts/average_checkpoints.py --inputs checkpoints/transformer \
   --num-epoch-checkpoints 10 --output checkpoints/transformer/model.pt

# Generate:
$ fairseq-generate data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/transformer/model.pt \
  --batch-size 128 --beam 5 --remove-bpe

```


### prepare-wmt14en2de.sh

The WMT English to German dataset can be preprocessed using the `prepare-wmt14en2de.sh` script.
By default it will produce a dataset that was modeled after ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), but with news-commentary-v12 data from WMT'17.

To use only data available in WMT'14 or to replicate results obtained in the original ["Convolutional Sequence to Sequence Learning" (Gehring et al., 2017)](https://arxiv.org/abs/1705.03122) paper, please use the `--icml17` option.

```
$ bash prepare-wmt14en2de.sh --icml17
```

Example usage:

```
$ cd examples/translation/
$ bash prepare-wmt14en2de.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/wmt17_en_de
$ fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1500 instead
$ mkdir -p checkpoints/fconv_wmt_en_de
$ fairseq-train data-bin/wmt17_en_de \
  --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --arch fconv_wmt_en_de --save-dir checkpoints/fconv_wmt_en_de

# Generate:
$ fairseq-generate data-bin/wmt17_en_de \
  --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt --beam 5 --remove-bpe

```

### prepare-wmt14en2fr.sh

Provides an example of pre-processing for the WMT'14 English to French translation task.

Example usage:

```
$ cd examples/translation/
$ bash prepare-wmt14en2fr.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/wmt14_en_fr
$ fairseq-preprocess --source-lang en --target-lang fr \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1000 instead
$ mkdir -p checkpoints/fconv_wmt_en_fr
$ fairseq-train data-bin/wmt14_en_fr \
  --lr 0.5 --clip-norm 0.1 --dropout 0.1 --max-tokens 3000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --arch fconv_wmt_en_fr --save-dir checkpoints/fconv_wmt_en_fr

# Generate:
$ fairseq-generate data-bin/fconv_wmt_en_fr \
  --path checkpoints/fconv_wmt_en_fr/checkpoint_best.pt --beam 5 --remove-bpe

```
