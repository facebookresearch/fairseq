# Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)
https://arxiv.org/pdf/1911.02116.pdf

# Larger-Scale Transformers for Multilingual Masked Language Modeling
https://arxiv.org/pdf/2105.00572.pdf


## What's New:
- June 2021: `XLMR-XL` AND `XLMR-XXL` models released.

## Introduction

`XLM-R` (`XLM-RoBERTa`) is a generic cross lingual sentence encoder that obtains state-of-the-art results on many cross-lingual understanding (XLU) benchmarks. It is trained on `2.5T` of filtered CommonCrawl data in 100 languages (list below).

 Language | Language|Language |Language | Language
---|---|---|---|---
Afrikaans | Albanian | Amharic | Arabic | Armenian 
Assamese | Azerbaijani | Basque | Belarusian | Bengali 
Bengali Romanize | Bosnian | Breton | Bulgarian | Burmese 
Burmese zawgyi font | Catalan | Chinese (Simplified) | Chinese (Traditional) | Croatian 
Czech | Danish | Dutch | English | Esperanto 
Estonian | Filipino | Finnish | French | Galician
Georgian | German | Greek | Gujarati | Hausa
Hebrew | Hindi | Hindi Romanize | Hungarian | Icelandic
Indonesian | Irish | Italian | Japanese | Javanese
Kannada | Kazakh | Khmer | Korean | Kurdish (Kurmanji)
Kyrgyz | Lao | Latin | Latvian | Lithuanian
Macedonian | Malagasy | Malay | Malayalam | Marathi
Mongolian | Nepali | Norwegian | Oriya | Oromo
Pashto | Persian | Polish | Portuguese | Punjabi
Romanian | Russian | Sanskrit | Scottish Gaelic | Serbian
Sindhi | Sinhala | Slovak | Slovenian | Somali
Spanish | Sundanese | Swahili | Swedish | Tamil
Tamil Romanize | Telugu | Telugu Romanize | Thai | Turkish
Ukrainian | Urdu | Urdu Romanize | Uyghur | Uzbek
Vietnamese | Welsh | Western Frisian | Xhosa | Yiddish

## Pre-trained models

Model | Description | #params | vocab size | Download
---|---|---|---|---
`xlmr.base` | XLM-R using the BERT-base architecture | 250M | 250k | [xlm.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz)
`xlmr.large` | XLM-R using the BERT-large architecture | 560M | 250k | [xlm.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz)
`xlmr.xl` | XLM-R (`layers=36, model_dim=2560`) | 3.5B | 250k | [xlm.xl.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr/xlmr.xl.tar.gz)
`xlmr.xxl` | XLM-R (`layers=48, model_dim=4096`) | 10.7B | 250k | [xlm.xxl.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr/xlmr.xxl.tar.gz)

## Results

**[XNLI (Conneau et al., 2018)](https://arxiv.org/abs/1809.05053)**

Model | average | en | fr | es | de | el | bg | ru | tr | ar | vi | th | zh | hi | sw | ur
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
`roberta.large.mnli` _(TRANSLATE-TEST)_ | 77.8 | 91.3 | 82.9 | 84.3 | 81.2 | 81.7 | 83.1 | 78.3 | 76.8 | 76.6 | 74.2 | 74.1 | 77.5 | 70.9 | 66.7 | 66.8
`xlmr.large` _(TRANSLATE-TRAIN-ALL)_ | 83.6 | 89.1 | 85.1 | 86.6 | 85.7 | 85.3 | 85.9 | 83.5 | 83.2 | 83.1 | 83.7 | 81.5 | 83.7 | 81.6 | 78.0 | 78.1
`xlmr.xl` _(TRANSLATE-TRAIN-ALL)_ | 85.4 | 91.1 | 87.2 | 88.1 | 87.0 | 87.4 | 87.8 | 85.3 | 85.2 | 85.3 | 86.2 | 83.8 | 85.3 | 83.1 | 79.8 | 78.2 | 85.4
`xlmr.xxl` _(TRANSLATE-TRAIN-ALL)_ | 86.0 | 91.5 | 87.6 | 88.7 | 87.8 | 87.4 | 88.2 | 85.6 | 85.1 | 85.8 | 86.3 | 83.9 | 85.6 | 84.6 | 81.7 | 80.6

**[MLQA (Lewis et al., 2018)](https://arxiv.org/abs/1910.07475)**

Model | average | en | es | de | ar | hi | vi | zh
---|---|---|---|---|---|---|---|---
`BERT-large` | - | 80.2/67.4 | - | - | - | - | - | -
`mBERT` | 57.7 / 41.6 | 77.7 / 65.2 | 64.3 / 46.6 | 57.9 / 44.3 | 45.7 / 29.8| 43.8 / 29.7 | 57.1 / 38.6 | 57.5 / 37.3
`xlmr.large` | 70.7 / 52.7 | 80.6 / 67.8 | 74.1 / 56.0 | 68.5 / 53.6 | 63.1 / 43.5 | 69.2 / 51.6 | 71.3 / 50.9 | 68.0 / 45.4
`xlmr.xl` | 73.4 / 55.3 | 85.1 / 72.6 | 66.7 / 46.2 | 70.5 / 55.5 | 74.3 / 56.9 | 72.2 / 54.7 | 74.4 / 52.9 | 70.9 / 48.5
`xlmr.xxl` | 74.8 / 56.6 | 85.5 / 72.4 | 68.6 / 48.4 | 72.7 / 57.8 | 75.4 / 57.6 | 73.7 / 55.8 | 76.0 / 55.0 | 71.7 / 48.9 


## Example usage

##### Load XLM-R from torch.hub (PyTorch >= 1.1):
```python
import torch
xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load XLM-R (for PyTorch 1.0 or custom models):
```python
# Download xlmr.large model
wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz
tar -xzvf xlmr.large.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import XLMRModel
xlmr = XLMRModel.from_pretrained('/path/to/xlmr.large', checkpoint_file='model.pt')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Apply sentence-piece-model (SPM) encoding to input text:
```python
en_tokens = xlmr.encode('Hello world!')
assert en_tokens.tolist() == [0, 35378,  8999, 38, 2]
xlmr.decode(en_tokens)  # 'Hello world!'

zh_tokens = xlmr.encode('你好，世界')
assert zh_tokens.tolist() == [0, 6, 124084, 4, 3221, 2]
xlmr.decode(zh_tokens)  # '你好，世界'

hi_tokens = xlmr.encode('नमस्ते दुनिया')
assert hi_tokens.tolist() == [0, 68700, 97883, 29405, 2]
xlmr.decode(hi_tokens)  # 'नमस्ते दुनिया'

ar_tokens = xlmr.encode('مرحبا بالعالم')
assert ar_tokens.tolist() == [0, 665, 193478, 258, 1705, 77796, 2]
xlmr.decode(ar_tokens) # 'مرحبا بالعالم'

fr_tokens = xlmr.encode('Bonjour le monde')
assert fr_tokens.tolist() == [0, 84602, 95, 11146, 2]
xlmr.decode(fr_tokens) # 'Bonjour le monde'
```

##### Extract features from XLM-R:
```python
# Extract the last layer's features
last_layer_features = xlmr.extract_features(zh_tokens)
assert last_layer_features.size() == torch.Size([1, 6, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = xlmr.extract_features(zh_tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
```

## Citation

```bibtex
@article{conneau2019unsupervised,
  title={Unsupervised Cross-lingual Representation Learning at Scale},
  author={Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and Zettlemoyer, Luke and Stoyanov, Veselin},
  journal={arXiv preprint arXiv:1911.02116},
  year={2019}
}
```


```bibtex
@article{goyal2021larger,
  title={Larger-Scale Transformers for Multilingual Masked Language Modeling},
  author={Goyal, Naman and Du, Jingfei and Ott, Myle and Anantharaman, Giri and Conneau, Alexis},
  journal={arXiv preprint arXiv:2105.00572},
  year={2021}
}
```
