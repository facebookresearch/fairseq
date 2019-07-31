# WMT 19

This page provides pointers to the models of Facebook-FAIR's WMT'19 news translation task submission [(Ng et al., 2019)](https://arxiv.org/abs/1907.06616).

## Pre-trained models

Description | Model
---|---
En->De Ensemble | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.bz2)
De->En Ensemble | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.bz2)
En->Ru Ensemble | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.bz2)
Ru->En Ensemble | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.bz2)
En LM | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2)
De LM | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2)
Ru LM | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2)

## Example usage (torch.hub)

```
>>> import torch
>>> en2de = torch.hub.load(
...   'pytorch/fairseq',
...   'transformer.wmt19.en-de',
...   checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt'
...   tokenizer='moses',
...   bpe='fastbpe',
... )
>>> en2de.generate("Machine learning is great!")
'Maschinelles Lernen ist großartig!'

>>> de2en = torch.hub.load(
...   'pytorch/fairseq',
...   'transformer.wmt19.de-en',
...   checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt'
...   tokenizer='moses',
...   bpe='fastbpe',
... )
>>> de2en.generate("Maschinelles Lernen ist großartig!")
'Machine learning is great!'

>>> en2ru = torch.hub.load(
...   'pytorch/fairseq',
...   'transformer.wmt19.en-ru',
...   checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt'
...   tokenizer='moses',
...   bpe='fastbpe',
... )
>>> en2ru.generate("Machine learning is great!")
'Машинное обучение - это здорово!'

>>> ru2en = torch.hub.load(
...   'pytorch/fairseq',
...   'transformer.wmt19.ru-en',
...   checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt'
...   tokenizer='moses',
...   bpe='fastbpe',
... )
>>> ru2en.generate("Машинное обучение - это здорово!")
'Machine learning is great!'

>>> en_lm = torch.hub.load(
...   'pytorch.fairseq',
...   'transformer_lm.wmt19.en'
...   tokenizer='moses',
...   bpe='fastbpe',
... )
>>> en_lm.generate("Machine learning is")
'Machine learning is the future of computing, says Microsoft boss Satya Nadella ...'

>>> de_lm = torch.hub.load(
...   'pytorch.fairseq',
...   'transformer_lm.wmt19.de'
...   tokenizer='moses',
...   bpe='fastbpe',
... )
>>> de_lm.generate("Maschinelles lernen ist")
''Maschinelles lernen ist das A und O (neues-deutschland.de) Die Arbeitsbedingungen für Lehrerinnen und Lehrer sind seit Jahren verbesserungswürdig ...'

>>> ru_lm = torch.hub.load(
...   'pytorch.fairseq',
...   'transformer_lm.wmt19.ru'
...   tokenizer='moses',
...   bpe='fastbpe',
... )
>>> ru_lm.generate("машинное обучение это")
'машинное обучение это то, что мы называем "искусственным интеллектом".'
```

## Citation
```bibtex
@inproceedings{ng2019facebook},
  title = {Facebook FAIR's WMT19 News Translation Task Submission},
  author = {Ng, Nathan and Yee, Kyra and Baevski, Alexei and Ott, Myle and Auli, Michael and Edunov, Sergey},
  booktitle = {Proc. of WMT},
  year = 2019,
}
```
