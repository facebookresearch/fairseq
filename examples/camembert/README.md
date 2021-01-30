# CamemBERT: a Tasty French Language Model

## Introduction

[CamemBERT](https://arxiv.org/abs/1911.03894) is a pretrained language model trained on 138GB of French text based on RoBERTa.

Also available in [github.com/huggingface/transformers](https://github.com/huggingface/transformers/).

## Pre-trained models

| Model                          | #params | Download                                                                                                                 | Arch. | Training data                     |
|--------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------|-------|-----------------------------------|
| `camembert` / `camembert-base` | 110M    | [camembert-base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base.tar.gz)                             | Base  | OSCAR (138 GB of text)            |
| `camembert-large`              | 335M    | [camembert-large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-large.tar.gz)                           | Large | CCNet (135 GB of text)            |
| `camembert-base-ccnet`         | 110M    | [camembert-base-ccnet.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-ccnet.tar.gz)                 | Base  | CCNet (135 GB of text)            |
| `camembert-base-wikipedia-4gb` | 110M    | [camembert-base-wikipedia-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-wikipedia-4gb.tar.gz) | Base  | Wikipedia (4 GB of text)          |
| `camembert-base-oscar-4gb`     | 110M    | [camembert-base-oscar-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-oscar-4gb.tar.gz)         | Base  | Subsample of OSCAR (4 GB of text) |
| `camembert-base-ccnet-4gb`     | 110M    | [camembert-base-ccnet-4gb.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/camembert-base-ccnet-4gb.tar.gz)         | Base  | Subsample of CCNet (4 GB of text) |

## Example usage

### fairseq
##### Load CamemBERT from torch.hub (PyTorch >= 1.1):
```python
import torch
camembert = torch.hub.load('pytorch/fairseq', 'camembert')
camembert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load CamemBERT (for PyTorch 1.0 or custom models):
```python
# Download camembert model
wget https://dl.fbaipublicfiles.com/fairseq/models/camembert-base.tar.gz
tar -xzvf camembert.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import CamembertModel
camembert = CamembertModel.from_pretrained('/path/to/camembert')
camembert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Filling masks:
```python
masked_line = 'Le camembert est <mask> :)'
camembert.fill_mask(masked_line, topk=3)
# [('Le camembert est délicieux :)', 0.4909118115901947, ' délicieux'),
#  ('Le camembert est excellent :)', 0.10556942224502563, ' excellent'),
#  ('Le camembert est succulent :)', 0.03453322499990463, ' succulent')]
```

##### Extract features from Camembert:
```python
# Extract the last layer's features
line = "J'aime le camembert !"
tokens = camembert.encode(line)
last_layer_features = camembert.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 10, 768])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
```

## Citation
If you use our work, please cite:

```bibtex
@inproceedings{martin2020camembert,
  title={CamemBERT: a Tasty French Language Model},
  author={Martin, Louis and Muller, Benjamin and Su{\'a}rez, Pedro Javier Ortiz and Dupont, Yoann and Romary, Laurent and de la Clergerie, {\'E}ric Villemonte and Seddah, Djam{\'e} and Sagot, Beno{\^\i}t},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
