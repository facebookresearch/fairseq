# RoBERTa: A Robustly Optimized BERT Pretraining Approach

https://arxiv.org/abs/1907.11692

## Introduction

**RoBERTa** iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. See the associated paper for more details.

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`roberta.base` | RoBERTa using the BERT-base architecture | 125M | [roberta.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)
`roberta.large` | RoBERTa using the BERT-large architecture | 355M | [roberta.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)
`roberta.large.mnli` | `roberta.large` finetuned on MNLI | 355M | [roberta.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz)

## Results

##### Results on GLUE tasks (dev set, single model, single-task finetuning)

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`roberta.base` | 87.6 | 92.8 | 91.9 | 78.7 | 94.8 | 90.2 | 63.6 | 91.2
`roberta.large` | 90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4
`roberta.large.mnli` | 90.2 | - | - | - | - | - | - | -

##### Results on SQuAD (dev set)

Model | SQuAD 1.1 EM/F1 | SQuAD 2.0 EM/F1
---|---|---
`roberta.large` | 88.9/94.6 | 86.5/89.4

##### Results on Reading Comprehension (RACE, test set)

Model | Accuracy | Middle | High
---|---|---|---
`roberta.large` | 83.2 | 86.5 | 81.3

## Example usage

##### Load RoBERTa from torch.hub (PyTorch >= 1.1):
```python
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load RoBERTa (for PyTorch 1.0):
```python
# Download roberta.large model
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('/path/to/roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Apply Byte-Pair Encoding (BPE) to input text:
```python
tokens = roberta.encode('Hello world!')
assert tokens.tolist() == [0, 31414, 232, 328, 2]
roberta.decode(tokens)  # 'Hello world!'
```

##### Extract features from RoBERTa:
```python
# Extract the last layer's features
last_layer_features = roberta.extract_features(tokens)
assert last_layer_features.size() == torch.Size([1, 5, 1024])

# Extract all layer's features (layer 0 is the embedding layer)
all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
assert len(all_layers) == 25
assert torch.all(all_layers[-1] == last_layer_features)
```

##### Use RoBERTa for sentence-pair classification tasks:
```python
# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

# Encode a pair of sentences and make a prediction
tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.')
roberta.predict('mnli', tokens).argmax()  # 0: contradiction

# Encode another pair of sentences
tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
roberta.predict('mnli', tokens).argmax()  # 2: entailment
```

##### Register a new (randomly initialized) classification head:
```python
roberta.register_classification_head('new_task', num_classes=3)
logprobs = roberta.predict('new_task', tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
```

##### Batched prediction:
```python
from fairseq.data.data_utils import collate_tokens
sentences = ['Hello world.', 'Another unrelated sentence.']
batch = collate_tokens([roberta.encode(sent) for sent in sentences], pad_idx=1)
logprobs = roberta.predict('new_task', batch)
assert logprobs.size() == torch.Size([2, 3])
```

##### Using the GPU:
```python
roberta.cuda()
roberta.predict('new_task', tokens)  # tensor([[-1.1050, -1.0672, -1.1245]], device='cuda:0', grad_fn=<LogSoftmaxBackward>)
```

##### Evaluating the `roberta.large.mnli` model

Example python code snippet to evaluate accuracy on the MNLI dev_matched set.
```python
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mnli', tokens).argmax().item()
        prediction_label = label_map[prediction]
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
# Expected output: 0.9060
```


## Finetuning

- [Finetuning on GLUE](README.finetune_glue.md)
- [Finetuning on custom classification tasks (e.g., IMDB)](README.finetune_custom_classification.md)
- Finetuning on SQuAD: coming soon

## Pretraining using your own data

You can use the [`masked_lm` task](/fairseq/tasks/masked_lm.py) to pretrain RoBERTa from scratch, or to continue pretraining RoBERTa starting from one of the released checkpoints.

Data should be preprocessed following the [language modeling example](/examples/language_model).

A more detailed tutorial is coming soon.

## Citation

```bibtex
@article{liu2019roberta,
    title = {RoBERTa: A Robustly Optimized BERT Pretraining Approach},
    author = {Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and
              Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and
              Luke Zettlemoyer and Veselin Stoyanov},
    journal={arXiv preprint arXiv:1907.11692},
    year = {2019},
}
```
