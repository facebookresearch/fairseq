# Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)

## Introduction

XLM-R (XLM-RoBERTa) is scaled cross lingual sentence encoder. It is trained on `2.5T` of data across `100` languages data filtered from Common Crawl. XLM-R achieves state-of-the-arts results on multiple cross lingual benchmarks.

## Pre-trained models

Model | Description | #params | vocab size | Download
---|---|---|---|---
`xlmr.base.v0` | XLM-R using the BERT-base architecture | 250M | 250k | [xlm.base.v0.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.v0.tar.gz)
`xlmr.large.v0` | XLM-R using the BERT-large architecture | 560M | 250k | [xlm.large.v0.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.v0.tar.gz)

(Note: The above models are still under training, we will update the weights, once fully trained, the results are based on the above checkpoints.)

## Results

**[XNLI (Conneau et al., 2018)](https://arxiv.org/abs/1809.05053)**

Model | average | en | fr | es | de | el | bg | ru | tr | ar | vi | th | zh | hi | sw | ur
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
`roberta.large.mnli` _(TRANSLATE-TEST)_ | 77.8 | 91.3 | 82.9 | 84.3 | 81.2 | 81.7 | 83.1 | 78.3 | 76.8 | 76.6 | 74.2 | 74.1 | 77.5 | 70.9 | 66.7 | 66.8
`xlmr.large.v0` _(TRANSLATE-TRAIN-ALL)_ | **82.4** | 88.7 | 85.2 | 85.6 | 84.6 | 83.6 | 85.5 | 82.4 | 81.6 | 80.9 | 83.4 | 80.9 | 83.3 | 79.8 | 75.9 | 74.3

**[MLQA (Lewis et al., 2018)](https://arxiv.org/abs/1910.07475)**

Model | average | en | es | de | ar | hi | vi | zh
---|---|---|---|---|---|---|---|---
`BERT-large` | - | 80.2/67.4 | - | - | - | - | - | -
`mBERT` | 57.7 / 41.6 | 77.7 / 65.2 | 64.3 / 46.6 | 57.9 / 44.3 | 45.7 / 29.8| 43.8 / 29.7 | 57.1 / 38.6 | 57.5 / 37.3
`xlmr.large.v0` | **70.0 / 52.2** | 80.1 / 67.7 | 73.2 / 55.1 | 68.3 / 53.7 | 62.8 / 43.7 | 68.3 / 51.0 | 70.5 / 50.1 | 67.1 / 44.4


## Example usage

##### Load XLM-R from torch.hub (PyTorch >= 1.1):
```python
import torch
xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large.v0')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load XLM-R (for PyTorch 1.0 or custom models):
```python
# Download xlmr.large model
wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.v0.tar.gz
tar -xzvf xlmr.large.v0.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import XLMRModel
xlmr = XLMRModel.from_pretrained('/path/to/xlmr.large.v0', checkpoint_file='model.pt')
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
