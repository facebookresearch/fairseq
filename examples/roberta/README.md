# RoBERTa: A Robustly Optimized BERT Pretraining Approach

*Pre-print coming 7/28*

## Introduction

**RoBERTa** iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. See the associated paper for more details.

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`roberta.base` | RoBERTa using the BERT-base architecture | 125M | [roberta.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)
`roberta.large` | RoBERTa using the BERT-large architecture | 355M | [roberta.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)
`roberta.large.mnli` | `roberta.large` finetuned on MNLI | 355M | [roberta.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz)

## Example usage (torch.hub)

##### Load RoBERTa:
```
>>> import torch
>>> roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
>>> roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Apply Byte-Pair Encoding (BPE) to input text:
```
>>> tokens = roberta.encode('Hello world!')
>>> tokens
tensor([    0, 31414,   232,   328,     2])
```

##### Extract features from RoBERTa:
```
>>> last_layer_features = roberta.extract_features(tokens)
>>> last_layer_features.size()
torch.Size([1, 5, 1024])

>>> all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
>>> len(all_layers)
25

>>> torch.all(all_layers[-1] == last_layer_features)
tensor(1, dtype=torch.uint8)
```

##### Use RoBERTa for sentence-pair classification tasks:
```
>>> roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')  # already finetuned
>>> roberta.eval()  # disable dropout for evaluation

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

##### Register a new (randomly initialized) classification head:
```
>>> roberta.register_classification_head('new_task', num_classes=3)
>>> roberta.predict('new_task', tokens)
tensor([[-1.1050, -1.0672, -1.1245]], grad_fn=<LogSoftmaxBackward>)
```

##### Using the GPU:
```
>>> roberta.cuda()
>>> roberta.predict('new_task', tokens)
tensor([[-1.1050, -1.0672, -1.1245]], device='cuda:0', grad_fn=<LogSoftmaxBackward>)
```

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

## Evaluating the `roberta.large.mnli` model

Example python code snippet to evaluate accuracy on the MNLI dev_matched set.
```
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

## Finetuning on GLUE tasks

A more detailed tutorial is coming soon.

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
  year = {2019},
}
```
