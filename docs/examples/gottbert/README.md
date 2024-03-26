# GottBERT: a pure German language model

## Introduction

[GottBERT](http://arxiv.org/abs/2012.02110) is a pretrained language model trained on 145GB of German text based on RoBERTa.

## Example usage

### fairseq
##### Load GottBERT from torch.hub (PyTorch >= 1.1):
```python
import torch
gottbert = torch.hub.load('pytorch/fairseq', 'gottbert-base')
gottbert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load GottBERT (for PyTorch 1.0 or custom models):
```python
# Download gottbert model
wget https://dl.gottbert.de/fairseq/models/gottbert-base.tar.gz
tar -xzvf gottbert.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import GottbertModel
gottbert = GottbertModel.from_pretrained('/path/to/gottbert')
gottbert.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Filling masks:
```python
masked_line = 'Gott ist <mask> ! :)'
gottbert.fill_mask(masked_line, topk=3)
# [('Gott ist gut ! :)',        0.3642110526561737,   ' gut'),
#  ('Gott ist überall ! :)',    0.06009674072265625,  ' überall'),
#  ('Gott ist großartig ! :)',  0.0370681993663311,   ' großartig')]
```

##### Extract features from GottBERT

```python
# Extract the last layer's features
line = "Der erste Schluck aus dem Becher der Naturwissenschaft macht atheistisch , aber auf dem Grunde des Bechers wartet Gott !"
tokens = gottbert.encode(line)
last_layer_features = gottbert.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 27, 768])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = gottbert.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 13
assert torch.all(all_layers[-1] == last_layer_features)
```
## Citation
If you use our work, please cite:

```bibtex
@misc{scheible2020gottbert,
      title={GottBERT: a pure German Language Model},
      author={Raphael Scheible and Fabian Thomczyk and Patric Tippmann and Victor Jaravine and Martin Boeker},
      year={2020},
      eprint={2012.02110},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
