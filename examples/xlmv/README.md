# XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models
https://arxiv.org/abs/2301.10472

## Introduction
 
`XLM-V` is a multilingual masked language model based on the `XLM-R` (`XLM-RoBERTa`) architecture with a 1M token vocabulary. It is trained on `2.5T` of filtered CommonCrawl data in 100 languages (list below).

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
`xlmv.base` | XLM-R style model with a 1M token vocabulary | 826M | 902K | [xlmv.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/xlmv/xlmv.base.tar.gz)

## Results

**[XNLI (Conneau et al., 2018)](https://arxiv.org/abs/1809.05053)**

Model | average | en | fr | es | de | el | bg | ru | tr | ar | vi | th | zh | hi | sw | ur
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
`xlmr.base` | 74.9 | 85.4 | 78.5 | 79.1 | 77.7 | 76.1 | 78.1 | 76.3 | 73.9 | 72.3 | 75.6 | 73.0 | 74.9 | 70.5 | 65.8 | 66.5 
`xlmv.base` | 76.0 | 85.6 | 79.6 | 79.5 | 78.4 | 76.9 | 79.6 | 76.6 | 74.0 | 73.1 | 76.2 | 73.0 | 75.1 | 72.0 | 70.5 | 69.4 

**[MLQA (Lewis et al., 2018)](https://arxiv.org/abs/1910.07475)**

Model | average | en | es | de | ar | hi | vi | zh
---|---|---|---|---|---|---|---|---
`xlmr.base` | 46.5 / 64.2 | 65.9 / 78.7 | 50.4 / 67.7 | 47.6 / 62.2 | 36.8 / 55.8 | 42.1 / 59.3 | 45.2 / 65.2 | 37.8 / 60.7
`xlmv.base` | 47.7 / 66.0 | 67.5 / 80.4 | 51.1 / 69.4 | 49.8 / 64.3 | 38.1 / 58.2 | 44.5 / 62.7 | 46.4 / 67.2 | 36.3 / 59.9


## Example usage

```python
# Download xlmv.base model
wget https://dl.fbaipublicfiles.com/fairseq/xlmv/xlmv.base.tar.gz
tar -xzvf xlmv.base.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import XLMRModel
xlmv = XLMRModel.from_pretrained('/path/to/xlmv.base', checkpoint_file='model.pt')
xlmv.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Apply sentence-piece-model (SPM) encoding to input text:
```python
en_tokens = xlmv.encode('Hello world!')
xlmv.decode(en_tokens)  # 'Hello world!'
```

##### Extract features from XLM-V:
```python
# Extract the last layer's features
last_layer_features = xlmv.extract_features(en_tokens)

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = xlmv.extract_features(en_tokens, return_all_hiddens=True)
```

## Citation

```bibtex
@article{liang2023xlmv,
  title={XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models},
  author={Liang, Davis and Gonen, Hila and Mao, Yuning and Hou, Rui and Goyal, Naman and Ghazvininejad, Marjan and Zettlemoyer, Luke and Khabsa, Madian},
  journal={arXiv preprint arxiv.org/abs/2301.10472},
  year={2023}
}
```
