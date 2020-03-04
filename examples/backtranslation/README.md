# Understanding Back-Translation at Scale (Edunov et al., 2018)

This page includes pre-trained models from the paper [Understanding Back-Translation at Scale (Edunov et al., 2018)](https://arxiv.org/abs/1808.09381).

## Pre-trained models

Model | Description | Dataset | Download
---|---|---|---
`transformer.wmt18.en-de` | Transformer <br> ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381)) <br> WMT'18 winner | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz) <br> See NOTE in the archive

## Example usage (torch.hub)

We require a few additional Python dependencies for preprocessing:
```bash
pip install subword_nmt sacremoses
```

Then to generate translations from the full model ensemble:
```python
import torch

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt18.en-de', ... ]

# Load the WMT'18 En-De ensemble
en2de_ensemble = torch.hub.load(
    'pytorch/fairseq', 'transformer.wmt18.en-de',
    checkpoint_file='wmt18.model1.pt:wmt18.model2.pt:wmt18.model3.pt:wmt18.model4.pt:wmt18.model5.pt',
    tokenizer='moses', bpe='subword_nmt')

# The ensemble contains 5 models
len(en2de_ensemble.models)
# 5

# Translate
en2de_ensemble.translate('Hello world!')
# 'Hallo Welt!'
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
