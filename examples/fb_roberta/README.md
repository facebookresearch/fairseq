# RoBERTa: A Robustly Optimized BERT Pretraining Approach

TODO: paper link

## Introduction

**RoBERTa** iterates on and optimizes BERT's pretraining strategy, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. See the associated paper for more detailed ablations demonstrating the impact of each of these changes.

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
>>> roberta.eval()

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
`roberta.base` | **87.6** | **92.8** | **91.9** | **78.7** | **94.8** | **90.2** | **63.6** | **91.2**

Model | MNLI-m | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`roberta.large` | **90.2** | **94.7** | **92.2** | **86.6** | **96.4** | **90.9** | **68.0** | **92.4**
`roberta.large.mnli` | **90.2** | - | - | - | - | - | - | -

### Results on SQuAD (dev set)

Model | SQuAD 1.1 EM/F1 | SQuAD 2.0 EM/F1
---|---|---
`roberta.large` | 88.9/94.6 | **86.5**/89.4

### Results on Reading Comprehension (RACE, test set)

Model | Accuracy | Middle | High
---|---|---|---
`roberta.large` | **83.2** | **86.5** | **81.3**

## Evaluating the `roberta.large.mnli` model

Example python code snippet to eval on MNLI dev_matched set.
```
label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
roberta.eval()
with open('glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, label = tokens[8], tokens[9], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('mnli', tokens).argmax().item()
        prediciton_label = label_map[prediction]
        ncorrect += int(prediciton_label == label)
        samples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
# Expected output: 0.9060
```

## Finetuning on GLUE tasks

Coming soon.

## Pretraining over your own data

Coming soon.

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
