# Understanding Back-Translation at Scale (Edunov et al., 2018)

This page includes pre-trained models from the paper [Understanding Back-Translation at Scale (Edunov et al., 2018)](https://arxiv.org/abs/1808.09381).

## Pre-trained models

Description | Dataset | Model | Test set(s)
---|---|---|---
Transformer <br> ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381); WMT'18 winner) | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz) | See NOTE in the archive

## Example usage

Interactive generation from the full ensemble via PyTorch Hub:
```
>>> import torch
>>> en2de_ensemble = torch.hub.load(
...   'pytorch/fairseq',
...   'transformer',
...   model_name_or_path='transformer.wmt18.en-de',
...   checkpoint_file='wmt18.model1.pt:wmt18.model2.pt:wmt18.model3.pt:wmt18.model4.pt:wmt18.model5.pt',
...   data_name_or_path='.',
...   tokenizer='moses',
...   aggressive_dash_splits=True,
...   bpe='subword_nmt',
... )
>>> len(en2de_ensemble.models)
5
>>> print(en2de_ensemble.generate('Hello world!'))
Hallo Welt!
```

## Citation
```bibtex
@inproceedings{edunov2018backtranslation,
  title = {Understanding Back-Translation at Scale},
  author = {Edunov, Sergey and Ott, Myle and Auli, Michael and Grangier, David},
  booktitle = {Conference of the Association for Computational Linguistics (ACL)},
  year = 2018,
}
```
