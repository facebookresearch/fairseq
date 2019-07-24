# RoBERTa: A Robustly Optimized BERT Pretraining Approach

TODO: paper link

## Introduction

**RoBERTa** iterates on and optimizes BERT's pretraining strategy, including removing the next-sentence prediction loss, and training with much larger mini-batches and higher learning rates. RoBERTa is additionally trained much longer using an order of magnitude more data than BERT. See the associated paper for detailed ablations demonstrating the impact of each of these changes.

## Pre-trained models

Model | Description | Download
---|---|---
`roberta.base` | RoBERTa using the BERT-base architecture | [roberta.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)
`roberta.large` | RoBERTa using the BERT-large architecture | [roberta.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)
`roberta.large.mnli` | `roberta.large` finetuned on MNLI | [roberta.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz)

## Example usage (torch.hub)

Load RoBERTa:
```
>>> import torch
>>> roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
```

Apply Byte-Pair Encoding (BPE) to input text:
```
>>> tokens = roberta.encode('Hello world!')
>>> tokens
tensor([    0, 31414,   232,   328,     2])
```

Extract features from RoBERTa:
```
>>> features = roberta.extract_features(tokens)
>>> features.size()
torch.Size([1, 5, 1024])
```

Use RoBERTa for sentence-pair classification tasks:
```
>>> roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')  # already finetuned

>>> tokens = roberta.encode(
...   'Roberta is a heavily optimized version of BERT.',
...   'Roberta is not very optimized.'
... )

>>> roberta.predict('mnli', tokens).argmax()
tensor(0)  # contradiction

>>> tokens = roberta.encode(
...   'Roberta is a heavily optimized version of BERT.',
...   'Roberta is based on BERT.'
... )

>>> roberta.predict('mnli', tokens).argmax()
tensor(2)  # entailment
```

Register a new (randomly initialized) classification head:
```
>>> roberta.register_classification_head('new_task', num_classes=3)
>>> roberta.predict('new_task', tokens)
tensor([[-1.2268, -1.1885, -0.9111]], grad_fn=<LogSoftmaxBackward>)
```

Using the GPU:
```
>>> roberta.cuda()
>>> roberta.predict('new_task', tokens)
tensor([[-1.2115, -1.1883, -0.9225]], device='cuda:0', grad_fn=<LogSoftmaxBackward>)
```

## Results

### Results on GLUE tasks (dev set, single model, no multi-task)

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
XLNet-base | 86.8 | 91.7 | 91.4 | 74.0 | 94.7 | 88.2 | 60.2 | 89.5
`roberta.base` | **87.6** | **92.8** | **91.9** | **78.7** | **94.8** | **90.2** | **63.6** | **91.2**

Model | MNLI-m | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
BERT-large | 86.6 | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0
XLNet-large | 89.8 | 93.9 | 91.8 | 83.8 | 95.6 | 89.2 | 63.6 | 91.8
`roberta.large` | **90.2** | **94.7** | **92.2** | **86.6** | **96.4** | **90.9** | **68.0** | **92.4**
`roberta.large.mnli` | **90.2** | - | - | - | - | - | - | -

### Results on SQuAD (dev set)

Model | SQuAD 1.1 EM/F1 | SQuAD 2.0 EM/F1
---|---|---
XLNet-base | -/- | 80.2/-
XLNet-large | -/- | 86.1/-
`roberta.base` | -/- | -/-
`roberta.large` | 88.9/94.6 | **86.5**/89.4

### Results on Reading Comprehension (RACE, test set)

Model | Accuracy | Middle | High
---|---|---|---
BERT-large | 72.0 | 76.6 | 70.1
XLNet-large | 81.7 | 85.4 | 80.2
`roberta.large` | **83.2** | **86.5** | **81.3**

## Finetuning on GLUE tasks

## Pretraining over your own data

## Citation

```bibtex
@article{liu2019roberta,
  title = {RoBERTa: A Robustly Optimized BERT Pretraining Approach},
  author = {Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and
            Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and
            Luke Zettlemoyer and Veselin Stoyanov},
  year = {2019},
}
```
