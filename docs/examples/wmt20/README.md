# WMT 20

This page provides pointers to the models of Facebook-FAIR's WMT'20 news translation task submission [(Chen et al., 2020)](https://arxiv.org/abs/2011.08298).

## Single best MT models (after finetuning on part of WMT20 news dev set)

Model | Description | Download
---|---|---
`transformer.wmt20.ta-en` | Ta->En | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz)
`transformer.wmt20.en-ta` | En->Ta | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz)
`transformer.wmt20.iu-en.news` | Iu->En (News domain) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz)
`transformer.wmt20.en-iu.news` | En->Iu (News domain) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz)
`transformer.wmt20.iu-en.nh` | Iu->En (Nunavut Hansard domain) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz)
`transformer.wmt20.en-iu.nh` | En->Iu (Nunavut Hansard domain) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz)

## Language models
Model | Description | Download
---|---|---
`transformer_lm.wmt20.en` | En Language Model | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en.tar.gz)
`transformer_lm.wmt20.ta` | Ta Language Model | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta.tar.gz)
`transformer_lm.wmt20.iu.news` | Iu Language Model (News domain) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu.news.tar.gz)
`transformer_lm.wmt20.iu.nh` | Iu Language Model (Nunavut Hansard domain) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu.nh.tar.gz)

## Example usage (torch.hub)

#### Translation

```python
import torch

# English to Tamil translation
en2ta = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.en-ta')
en2ta.translate("Machine learning is great!")  # 'இயந்திரக் கற்றல் அருமை!'

# Tamil to English translation
ta2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.ta-en')
ta2en.translate("இயந்திரக் கற்றல் அருமை!")  # 'Machine learning is great!'

# English to Inuktitut translation
en2iu = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.en-iu.news')
en2iu.translate("machine learning is great!")  # 'ᖃᒧᑕᐅᔭᓄᑦ ᐃᓕᓐᓂᐊᕐᓂᖅ ᐱᐅᔪᒻᒪᕆᒃ!'

# Inuktitut to English translation
iu2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.iu-en.news')
iu2en.translate("ᖃᒧᑕᐅᔭᓄᑦ ᐃᓕᓐᓂᐊᕐᓂᖅ ᐱᐅᔪᒻᒪᕆᒃ!")  # 'Machine learning excellence!'
```

#### Language Modeling

```python
# Sample from the English LM
en_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt20.en')
en_lm.sample("Machine learning is")  # 'Machine learning is a type of artificial intelligence that uses machine learning to learn from data and make predictions.'

# Sample from the Tamil LM
ta_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt20.ta')
ta_lm.sample("இயந்திரக் கற்றல் என்பது செயற்கை நுண்ணறிவின்")  # 'இயந்திரக் கற்றல் என்பது செயற்கை நுண்ணறிவின் ஒரு பகுதியாகும்.'

# Sample from the Inuktitut LM
iu_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt20.iu.news')
iu_lm.sample("ᖃᒧᑕᐅᔭᓄᑦ ᐃᓕᓐᓂᐊᕐᓂᖅ")  # 'ᖃᒧᑕᐅᔭᓄᑦ ᐃᓕᓐᓂᐊᕐᓂᖅ, ᐊᒻᒪᓗ ᓯᓚᐅᑉ ᐊᓯᙳᖅᐸᓪᓕᐊᓂᖓᓄᑦ ᖃᓄᐃᓕᐅᕈᑎᒃᓴᑦ, ᐃᓚᖃᖅᖢᑎᒃ ᐅᑯᓂᖓ:'
```

## Citation
```bibtex
@inproceedings{chen2020facebook
  title={Facebook AI's WMT20 News Translation Task Submission},
  author={Peng-Jen Chen and Ann Lee and Changhan Wang and Naman Goyal and Angela Fan and Mary Williamson and Jiatao Gu},
  booktitle={Proc. of WMT},
  year={2020},
}
```
