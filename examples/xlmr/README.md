# Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)

## Introduction

XLM-R (XLM-RoBERTa) is scaled cross lingual sentence encoder. It is trained on `2.5T` of data across `100` languages data filtered from Common Crawl. XLM-R achieves state-of-the-arts results on multiple cross lingual benchmarks.

## Pre-trained models

Model | Description | #params | vocab size | Download
---|---|---|---|---
`xlmr.base` | XLM-R using the BERT-base architecture | 250M | 250k | [xlm.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz)
`xlmr.large` | XLM-R using the BERT-large architecture | 560M | 250k | [xlm.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz)

(Note: Above are final model checkpoints. If you were using previously released `v0` version, we recommend using above. They have same architecture and dictionary.)

## Results

**[XNLI (Conneau et al., 2018)](https://arxiv.org/abs/1809.05053)**

Model | average | en | fr | es | de | el | bg | ru | tr | ar | vi | th | zh | hi | sw | ur
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
`roberta.large.mnli` _(TRANSLATE-TEST)_ | 77.8 | 91.3 | 82.9 | 84.3 | 81.2 | 81.7 | 83.1 | 78.3 | 76.8 | 76.6 | 74.2 | 74.1 | 77.5 | 70.9 | 66.7 | 66.8
`xlmr.large` _(TRANSLATE-TRAIN-ALL)_ | **83.6** | 89.1 | 85.1 | 86.6 | 85.7 | 85.3 | 85.9 | 83.5 | 83.2 | 83.1 | 83.7 | 81.5 | 83.7 | 81.6 | 78.0 | 78.1

**[MLQA (Lewis et al., 2018)](https://arxiv.org/abs/1910.07475)**

Model | average | en | es | de | ar | hi | vi | zh
---|---|---|---|---|---|---|---|---
`BERT-large` | - | 80.2/67.4 | - | - | - | - | - | -
`mBERT` | 57.7 / 41.6 | 77.7 / 65.2 | 64.3 / 46.6 | 57.9 / 44.3 | 45.7 / 29.8| 43.8 / 29.7 | 57.1 / 38.6 | 57.5 / 37.3
`xlmr.large` | **70.7 / 52.7** | 80.6 / 67.8 | 74.1 / 56.0 | 68.5 / 53.6 | 63.1 / 43.5 | 69.2 / 51.6 | 71.3 / 50.9 | 68.0 / 45.4


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
@article{,
    title = {Unsupervised Cross-lingual Representation Learning at Scale},
    author = {Alexis Conneau and Kartikay Khandelwal
        and Naman Goyal and Vishrav Chaudhary and Guillaume Wenzek
        and Francisco Guzm\'an and Edouard Grave and Myle Ott
        and Luke Zettlemoyer and Veselin Stoyanov
    },
    journal={},
    year = {2019},
}
```
