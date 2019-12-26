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

We require a few additional Python dependencies for preprocessing:
```bash
pip install fastBPE sacremoses
```

To sample from a language model using PyTorch Hub:
```python
import torch

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer_lm.wmt19.en', ...]

# Load an English LM trained on WMT'19 News Crawl data
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', tokenizer='moses', bpe='fastbpe')
en_lm.eval()  # disable dropout

# Move model to GPU
en_lm.cuda()

# Sample from the language model
en_lm.sample('Barack Obama', beam=1, sampling=True, sampling_topk=10, temperature=0.8)
# "Barack Obama is coming to Sydney and New Zealand (...)"

# Compute perplexity for a sequence
en_lm.score('Barack Obama is coming to Sydney and New Zealand')['positional_scores'].mean().neg().exp()
# tensor(15.1474)

# The same interface can be used with custom models as well
from fairseq.models.transformer_lm import TransformerLanguageModel
custom_lm = TransformerLanguageModel.from_pretrained('/path/to/model/dir', 'checkpoint100.pt', tokenizer='moses', bpe='fastbpe')
custom_lm.sample('Barack Obama', beam=5)
# "Barack Obama (...)"
```

## Training a transformer language model with the CLI tools

### 1) Preprocess the data

First download and prepare the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/):
```bash
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..
```

Next preprocess/binarize the data:
```bash
TEXT=examples/language_model/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

### 2) Train a language model

Next we'll train a basic transformer language model on wikitext-103. For more
advanced examples (e.g., using [adaptive inputs](https://arxiv.org/abs/1809.10853)),
please see the [Transformer LM README](transformer_lm/README.md).

To train a basic LM (assumes 2 GPUs):
```
$ fairseq-train --task language_modeling \
  data-bin/wikitext-103 \
  --save-dir checkpoints/transformer_wikitext-103 \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --fp16 \
  --max-update 50000
```

If you run out of memory, try reducing `--max-tokens` (max number of tokens per
batch) or `--tokens-per-sample` (max sequence length). You can also adjust
`--update-freq` to accumulate gradients and simulate training on a different
number of GPUs.

### 3) Evaluate
```bash
fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/transformer_wiki103/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024
```

## Convolutional language models

Please see the [convolutional LM README](conv_lm/README.md) for instructions to
train convolutional language models.
