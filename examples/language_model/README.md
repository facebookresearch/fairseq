# Neural Language Modeling

## Pre-trained models

Model | Description | Dataset | Download
---|---|---|---
`transformer_lm.gbw.adaptive_huge` | Adaptive Inputs <br> ([Baevski and Auli, 2018](https://arxiv.org/abs/1809.10853)) <br> 1026M params | [Google Billion Words](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2)
`transformer_lm.wiki103.adaptive` | Adaptive Inputs <br> ([Baevski and Auli, 2018](https://arxiv.org/abs/1809.10853)) <br> 247M params | [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.tar.bz2)
`transformer_lm.wmt19.en` | English LM <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) | [WMT News Crawl](http://data.statmt.org/news-crawl/) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.gz)
`transformer_lm.wmt19.de` | German LM <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) | [WMT News Crawl](http://data.statmt.org/news-crawl/) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.gz)
`transformer_lm.wmt19.ru` | Russian LM <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) | [WMT News Crawl](http://data.statmt.org/news-crawl/) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.gz)

## Example usage

Sampling from a language model using PyTorch Hub:
```python
import torch

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer_lm.wmt19.en', ...]

# Load an English LM trained on WMT'19 News Crawl data
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', tokenizer='moses', bpe='fastbpe')

# Sample from the language model
en_lm.sample('Barack Obama', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
# "Barack Obama is coming to Sydney and New Zealand (...)"
```

## Training a new model with the CLI tools

These scripts provide an example of pre-processing data for the Language Modeling task.

### prepare-wikitext-103.sh

Provides an example of pre-processing for [WikiText-103 language modeling task](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/):

Example usage:

Prepare data:
```bash
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..

# Binarize the dataset:
TEXT=examples/language_model/wikitext-103

fairseq-preprocess --only-source \
    --trainpref $TEXT/wiki.train.tokens --validpref $TEXT/wiki.valid.tokens --testpref $TEXT/wiki.test.tokens \ 
    --destdir data-bin/wikitext-103
```

Train a transformer language model with adaptive inputs ([Baevski and Auli (2018): Adaptive Input Representations for Neural Language Modeling](transformer_lm/README.md)):
```bash
# If it runs out of memory, try to reduce max-tokens and tokens-per-sample
mkdir -p checkpoints/transformer_wikitext-103
fairseq-train --task language_modeling data-bin/wikitext-103 \
    --save-dir checkpoints/transformer_wikitext-103 --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d

# Evaluate:
fairseq-eval-lm data-bin/wikitext-103 --path 'checkpoints/transformer_wiki103/checkpoint_best.pt' \
    --sample-break-mode complete --max-tokens 3072 --context-window 2560 --softmax-batch 1024
```

Train a convolutional language model ([Dauphin et al. (2017): Language Modeling with Gated Convolutional Networks](conv_lm/README.md)):
```
# If it runs out of memory, try to reduce max-tokens and tokens-per-sample
mkdir -p checkpoints/fconv_wikitext-103
fairseq-train --task language_modeling data-bin/wikitext-103 \
    --save-dir checkpoints/fconv_wikitext-103 \
    --max-epoch 35 --arch fconv_lm_dauphin_wikitext103 --optimizer nag \
    --lr 1.0 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --clip-norm 0.1 --dropout 0.2 --weight-decay 5e-06 --criterion adaptive_loss \
    --adaptive-softmax-cutoff 10000,20000,200000 --max-tokens 1024 --tokens-per-sample 1024 \
    --ddp-backend=no_c10d

# Evaluate:
fairseq-eval-lm data-bin/wikitext-103 --path 'checkpoints/fconv_wiki103/checkpoint_best.pt'
```
