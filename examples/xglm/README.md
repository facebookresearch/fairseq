# Few-shot Learning with Multilingual Language Models

## Introduction

In this work, we train multilingual generative language models, dubbed XGLM, on a balanced corpus covering a diverse set of languages, and study their few- and zero-shot learning capabilities in a wide range of tasks. Our largest model with 7.5 billion parameters sets new state of the art in few-shot learning on more than 20 representative languages, outperforming GPT-3 of comparable size in multilingual commonsense reasoning (+7.4 accuracy points for 0-shot, +9.4 for 4-shot) and natural language inference (+5.4 for 0-shot, +5.4 for 4-shot). We have included a [model card](model_card.md) of XGLM for transparency and accountability.

## Data and Languages
XGLM models are trained on a new multilingual corpus extracted from CommonCrawl (CC100-XL), a significantly larger multilingual dataset covering 68 Common Crawl (CC) snapshots (from [Summer 2013](http://commoncrawl.org/2013/11/new-crawl-data-available/) to [March/April 2020](https://commoncrawl.org/2020/04/march-april-2020-crawl-archive-now-available/) consisting of 134 languages. The detailed languages and data statistics are reported in the paper (Table A.1).

## Pre-trained models

Model | Layers | Model Dim | Languages | Download
---|---|---|---|---
`XGLM 564M` | 24 | 1024 | trained on 30 languages|  [xglm.564M.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.564M.tar.gz)
`XGLM 1.7B` | 24 | 2048 | trained on 30 languages|  [xglm.1.7B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.1.7B.tar.gz)
`XGLM 2.9B` | 48 | 2048 | trained on 30 languages|  [xglm.2.9B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.2.9B.tar.gz)
`XGLM 7.5B` | 32 | 4096 | trained on 30 languages|  [xglm.7.5B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.7.5B.tar.gz)
`XGLM 4.5B` | 48 | 2048 | trained on 134 languages|  [xglm.4.5B.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.4.5B.tar.gz)

## Evaluation
Coming soon.

## Citation
Coming soon.
