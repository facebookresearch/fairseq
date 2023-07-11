# WMT 19

This page provides pointers to the models of Facebook-FAIR's WMT'19 news translation task submission [(Ng et al., 2019)](https://arxiv.org/abs/1907.06616).

## Pre-trained models

Model | Description | Download
---|---|---
`transformer.wmt19.en-de` | En->De Ensemble | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz)
`transformer.wmt19.de-en` | De->En Ensemble | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz)
`transformer.wmt19.en-ru` | En->Ru Ensemble | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz)
`transformer.wmt19.ru-en` | Ru->En Ensemble | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz)
`transformer_lm.wmt19.en` | En Language Model | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.gz)
`transformer_lm.wmt19.de` | De Language Model | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.gz)
`transformer_lm.wmt19.ru` | Ru Language Model | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.gz)

## Pre-trained single models before finetuning

Model | Description | Download
---|---|---
`transformer.wmt19.en-de` | En->De Single, no finetuning | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.ffn8192.tar.gz)
`transformer.wmt19.de-en` | De->En Single, no finetuning  | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.ffn8192.tar.gz)
`transformer.wmt19.en-ru` | En->Ru Single, no finetuning | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ffn8192.tar.gz)
`transformer.wmt19.ru-en` | Ru->En Single, no finetuning  | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ffn8192.tar.gz)

## Example usage (torch.hub)

#### Requirements

We require a few additional Python dependencies for preprocessing:
```bash
pip install fastBPE sacremoses
```

#### Translation

```python
import torch

# English to German translation
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
en2de.translate("Machine learning is great!")  # 'Maschinelles Lernen ist großartig!'

# German to English translation
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
de2en.translate("Maschinelles Lernen ist großartig!")  # 'Machine learning is great!'

# English to Russian translation
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
en2ru.translate("Machine learning is great!")  # 'Машинное обучение - это здорово!'

# Russian to English translation
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en', checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
ru2en.translate("Машинное обучение - это здорово!")  # 'Machine learning is great!'
```

#### Language Modeling

```python
# Sample from the English LM
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.en', tokenizer='moses', bpe='fastbpe')
en_lm.sample("Machine learning is")  # 'Machine learning is the future of computing, says Microsoft boss Satya Nadella ...'

# Sample from the German LM
de_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.de', tokenizer='moses', bpe='fastbpe')
de_lm.sample("Maschinelles lernen ist")  # 'Maschinelles lernen ist das A und O (neues-deutschland.de) Die Arbeitsbedingungen für Lehrerinnen und Lehrer sind seit Jahren verbesserungswürdig ...'

# Sample from the Russian LM
ru_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.ru', tokenizer='moses', bpe='fastbpe')
ru_lm.sample("машинное обучение это")  # 'машинное обучение это то, что мы называем "искусственным интеллектом".'
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
