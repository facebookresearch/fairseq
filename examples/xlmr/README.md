# Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa)

## Introduction

XLM-R (XLM-RoBERTa) is scaled cross lingual sentence encoder. It is trained on `2.5T` of data across `100` languages data filtered from Common Crawl. XLM-R achieves state-of-the-arts results on multiple cross lingual benchmarks.

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`xlmr.base.v0` | XLM-R using the BERT-base architecture | 250M | [xlm.base.v0.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.v0.tar.gz)
`xlmr.large.v0` | XLM-R using the BERT-large architecture | 560M | [xlm.large.v0.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.v0.tar.gz)

(Note: The above models are still under training, we will update the weights, once fully trained, the results are based on the above checkpoints.)

## Results

**[XNLI (Conneau et al., 2018)](https://arxiv.org/abs/1809.05053)**

Model | en | fr | es | de | el | bg | ru | tr | ar | vi | th | zh | hi | sw | ur
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
`roberta.large.mnli` _(TRANSLATE-TEST)_ | 91.3 | 82.9 | 84.3 | 81.24 | 81.74 | 83.13 | 78.28 | 76.79 | 76.64 | 74.17 | 74.05 | 77.5 | 70.9 | 66.65 | 66.81
`xlmr.large.v0` _(TRANSLATE-TRAIN-ALL)_ | 88.7 | 85.2 | 85.6 | 84.6 | 83.6 | 85.5 | 82.4 | 81.6 | 80.9 | 83.4 | 80.9 | 83.3 | 79.8 | 75.9 | 74.3

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

##### Apply Byte-Pair Encoding (BPE) to input text:
```python
tokens = xlmr.encode('Hello world!')
assert tokens.tolist() == [    0, 35378,  8999,    38,     2]
xlmr.decode(tokens)  # 'Hello world!'
```

##### Extract features from XLM-R:
```python
# Extract the last layer's features
last_layer_features = xlmr.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = xlmr.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
```

## Citation

```bibtex
@article{,
    title = {Unsupervised Cross-lingual Representation Learning at Scale},
    author = {Alexis Conneau and Kartikay Khandelwal and Naman Goyal
        and Vishrav Chaudhary and Guillaume Wenzek and Francisco Guzm\'an
        and Edouard Grave and Myle Ott and Luke Zettlemoyer and Veselin Stoyanov
    },
    journal={},
    year = {2019},
}
```
