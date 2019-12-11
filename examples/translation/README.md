# Neural Machine Translation

This README contains instructions for [using pretrained translation models](#example-usage-torchhub)
as well as [training new models](#training-a-new-model).

## Pre-trained models

Model | Description | Dataset | Download
---|---|---|---
`conv.wmt14.en-fr` | Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
`conv.wmt14.en-de` | Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-de.newstest2014.tar.bz2)
`conv.wmt17.en-de` | Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.v2.en-de.newstest2014.tar.bz2)
`transformer.wmt14.en-fr` | Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
`transformer.wmt16.en-de` | Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
`transformer.wmt18.en-de` | Transformer <br> ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381)) <br> WMT'18 winner | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz) <br> See NOTE in the archive
`transformer.wmt19.en-de` | Transformer <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) <br> WMT'19 winner | [WMT'19 English-German](http://www.statmt.org/wmt19/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz)
`transformer.wmt19.de-en` | Transformer <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) <br> WMT'19 winner | [WMT'19 German-English](http://www.statmt.org/wmt19/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz)
`transformer.wmt19.en-ru` | Transformer <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) <br> WMT'19 winner | [WMT'19 English-Russian](http://www.statmt.org/wmt19/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz)
`transformer.wmt19.ru-en` | Transformer <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) <br> WMT'19 winner | [WMT'19 Russian-English](http://www.statmt.org/wmt19/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz)

## Example usage (torch.hub)

We require a few additional Python dependencies for preprocessing:
```bash
pip install sacremoses subword_nmt
```

Interactive translation via PyTorch Hub:
```python
import torch

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

# Load a transformer trained on WMT'16 En-De
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')

# The underlying model is available under the *models* attribute
assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)

# Translate a sentence
en2de.translate('Hello world!')
# 'Hallo Welt!'
```

Loading custom models:
```python
from fairseq.models.transformer import TransformerModel
zh2en = TransformerModel.from_pretrained(
  '/path/to/checkpoints',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/wmt17_zh_en_full',
  bpe='subword_nmt',
  bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
)
zh2en.translate('你好 世界')
# 'Hello World'
```

## Example usage (CLI tools)

Generation with the binarized test sets can be run in batch mode as follows, e.g. for WMT 2014 English-French on a GTX-1080ti:
```bash
mkdir -p data-bin
curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
curl https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
fairseq-generate data-bin/wmt14.en-fr.newstest2014  \
    --path data-bin/wmt14.en-fr.fconv-py/model.pt \
    --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
# ...
# | Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
# | Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Compute BLEU score
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
# BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

## Training a new model

### IWSLT'14 German to English (Transformer)

The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

First download and preprocess the data:
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

Next we'll train a Transformer translation model over this data:
```bash
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096
```

Finally we can evaluate our trained model:
```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```

### WMT'14 English to German (Convolutional)

The following instructions can be used to train a Convolutional translation model on the WMT English to German dataset.
See the [Scaling NMT README](../scaling_nmt/README.md) for instructions to train a Transformer translation model on this data.

The WMT English to German dataset can be preprocessed using the `prepare-wmt14en2de.sh` script.
By default it will produce a dataset that was modeled after [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), but with additional news-commentary-v12 data from WMT'17.

To use only data available in WMT'14 or to replicate results obtained in the original [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](https://arxiv.org/abs/1705.03122) paper, please use the `--icml17` option.

```bash
# Download and prepare the data
cd examples/translation/
# WMT'17 data:
bash prepare-wmt14en2de.sh
# or to use WMT'14 data:
# bash prepare-wmt14en2de.sh --icml17
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

# Train the model
mkdir -p checkpoints/fconv_wmt_en_de
fairseq-train \
    data-bin/wmt17_en_de \
    --arch fconv_wmt_en_de \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/fconv_wmt_en_de

# Evaluate
fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

### WMT'14 English to French
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-wmt14en2fr.sh
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt14_en_fr
fairseq-preprocess \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60

# Train the model
mkdir -p checkpoints/fconv_wmt_en_fr
fairseq-train \
    data-bin/wmt14_en_fr \
    --lr 0.5 --clip-norm 0.1 --dropout 0.1 --max-tokens 3000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --arch fconv_wmt_en_fr \
    --save-dir checkpoints/fconv_wmt_en_fr

# Evaluate
fairseq-generate \
    data-bin/fconv_wmt_en_fr \
    --path checkpoints/fconv_wmt_en_fr/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

## Multilingual Translation

We also support training multilingual translation models. In this example we'll
train a multilingual `{de,fr}-en` translation model using the IWSLT'17 datasets.

Note that we use slightly different preprocessing here than for the IWSLT'14
En-De data above. In particular we learn a joint BPE code for all three
languages and use interactive.py and sacrebleu for scoring the test set.

```bash
# First install sacrebleu and sentencepiece
pip install sacrebleu sentencepiece

# Then download and preprocess the data
cd examples/translation/
bash prepare-iwslt17-multilingual.sh
cd ../..

# Binarize the de-en dataset
TEXT=examples/translation/iwslt17.de_fr.en.bpe16k
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en --validpref $TEXT/valid.bpe.de-en \
    --joined-dictionary \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Binarize the fr-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $TEXT/train.bpe.fr-en --validpref $TEXT/valid.bpe.fr-en \
    --joined-dictionary --tgtdict data-bin/iwslt17.de_fr.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Train a multilingual transformer model
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs
mkdir -p checkpoints/multilingual_transformer
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
    --max-epoch 50 \
    --ddp-backend=no_c10d \
    --task multilingual_translation --lang-pairs de-en,fr-en \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir checkpoints/multilingual_transformer \
    --max-tokens 4000 \
    --update-freq 8

# Generate and score the test set with sacrebleu
SRC=de
sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --echo src \
    | python scripts/spm_encode.py --model examples/translation/iwslt17.de_fr.en.bpe16k/sentencepiece.bpe.model \
    > iwslt17.test.${SRC}-en.${SRC}.bpe
cat iwslt17.test.${SRC}-en.${SRC}.bpe \
    | fairseq-interactive data-bin/iwslt17.de_fr.en.bpe16k/ \
      --task multilingual_translation --source-lang ${SRC} --target-lang en \
      --path checkpoints/multilingual_transformer/checkpoint_best.pt \
      --buffer-size 2000 --batch-size 128 \
      --beam 5 --remove-bpe=sentencepiece \
    > iwslt17.test.${SRC}-en.en.sys
grep ^H iwslt17.test.${SRC}-en.en.sys | cut -f3 \
    | sacrebleu --test-set iwslt17 --language-pair ${SRC}-en
```

##### Argument format during inference

During inference it is required to specify a single `--source-lang` and
`--target-lang`, which indicates the inference langauge direction.
`--lang-pairs`, `--encoder-langtok`, `--decoder-langtok` have to be set to
the same value as training.
