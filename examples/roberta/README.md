# RoBERTa: A Robustly Optimized BERT Pretraining Approach

https://arxiv.org/abs/1907.11692

## Introduction

RoBERTa iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data. See the associated paper for more details.

### What's New:

- August 2019: RoBERTa is now supported in the [pytorch-transformers library](https://github.com/huggingface/pytorch-transformers).
- August 2019: Added [tutorial for finetuning on WinoGrande](https://github.com/pytorch/fairseq/tree/master/examples/roberta/wsc#roberta-training-on-winogrande-dataset).
- August 2019: Added [tutorial for pretraining RoBERTa using your own data](README.pretraining.md).

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`roberta.base` | RoBERTa using the BERT-base architecture | 125M | [roberta.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)
`roberta.large` | RoBERTa using the BERT-large architecture | 355M | [roberta.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)
`roberta.large.mnli` | `roberta.large` finetuned on [MNLI](http://www.nyu.edu/projects/bowman/multinli) | 355M | [roberta.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz)
`roberta.large.wsc` | `roberta.large` finetuned on [WSC](wsc/README.md) | 355M | [roberta.large.wsc.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz)

## Results

**[GLUE (Wang et al., 2019)](https://gluebenchmark.com/)**
_(dev set, single model, single-task finetuning)_

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`roberta.base` | 87.6 | 92.8 | 91.9 | 78.7 | 94.8 | 90.2 | 63.6 | 91.2
`roberta.large` | 90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4
`roberta.large.mnli` | 90.2 | - | - | - | - | - | - | -

**[SuperGLUE (Wang et al., 2019)](https://super.gluebenchmark.com/)**
_(dev set, single model, single-task finetuning)_

Model | BoolQ | CB | COPA | MultiRC | RTE | WiC | WSC
---|---|---|---|---|---|---|---
`roberta.large` | 86.9 | 98.2 | 94.0 | 85.7 | 89.5 | 75.6 | -
`roberta.large.wsc` | - | - | - | - | - | - | 91.3

**[SQuAD (Rajpurkar et al., 2018)](https://rajpurkar.github.io/SQuAD-explorer/)**
_(dev set, no additional data used)_

Model | SQuAD 1.1 EM/F1 | SQuAD 2.0 EM/F1
---|---|---
`roberta.large` | 88.9/94.6 | 86.5/89.4

**[RACE (Lai et al., 2017)](http://www.qizhexie.com/data/RACE_leaderboard.html)**
_(test set)_

Model | Accuracy | Middle | High
---|---|---|---
`roberta.large` | 83.2 | 86.5 | 81.3

**[HellaSwag (Zellers et al., 2019)](https://rowanzellers.com/hellaswag/)**
_(test set)_

Model | Overall | In-domain | Zero-shot | ActivityNet | WikiHow
---|---|---|---|---|---
`roberta.large` | 85.2 | 87.3 | 83.1 | 74.6 | 90.9

**[Commonsense QA (Talmor et al., 2019)](https://www.tau-nlp.org/commonsenseqa)**
_(test set)_

Model | Accuracy
---|---
`roberta.large` (single model) | 72.1
`roberta.large` (ensemble) | 72.5

**[Winogrande (Sakaguchi et al., 2019)](https://arxiv.org/abs/1907.10641)**
_(test set)_

Model | Accuracy
---|---
`roberta.large` | 78.1

## Example usage

##### Load RoBERTa from torch.hub (PyTorch >= 1.1):
```python
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load RoBERTa (for PyTorch 1.0 or custom models):
```python
# Download roberta.large model
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('/path/to/roberta.large', checkpoint_file='model.pt')
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

## Advanced usage

#### Filling masks:

RoBERTa can be used to fill `<mask>` tokens in the input. Some examples from the
[Natural Questions dataset](https://ai.google.com/research/NaturalQuestions/):
```python
roberta.fill_mask('The first Star wars movie came out in <mask>', topk=3)
# [('The first Star wars movie came out in 1977', 0.9504712224006653), ('The first Star wars movie came out in 1978', 0.009986752644181252), ('The first Star wars movie came out in 1979', 0.00957468245178461)]

roberta.fill_mask('Vikram samvat calender is official in <mask>', topk=3)
# [('Vikram samvat calender is official in India', 0.21878768503665924), ('Vikram samvat calender is official in Delhi', 0.08547217398881912), ('Vikram samvat calender is official in Gujarat', 0.07556255906820297)]

roberta.fill_mask('<mask> is the common currency of the European Union', topk=3)
# [('Euro is the common currency of the European Union', 0.945650577545166), ('euro is the common currency of the European Union', 0.025747718289494514), ('€ is the common currency of the European Union', 0.011183015070855618)]
```

#### Pronoun disambiguation (Winograd Schema Challenge):

RoBERTa can be used to disambiguate pronouns. First install spaCy and download the English-language model:
```bash
pip install spacy
python -m spacy download en_core_web_lg
```

Next load the `roberta.large.wsc` model and call the `disambiguate_pronoun`
function. The pronoun should be surrounded by square brackets (`[]`) and the
query referent surrounded by underscores (`_`), or left blank to return the
predicted candidate text directly:
```python
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.wsc', user_dir='examples/roberta/wsc')
roberta.cuda()  # use the GPU (optional)

roberta.disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
# True
roberta.disambiguate_pronoun('The trophy would not fit in the brown _suitcase_ because [it] was too big.')
# False

roberta.disambiguate_pronoun('The city councilmen refused the demonstrators a permit because [they] feared violence.')
# 'The city councilmen'
roberta.disambiguate_pronoun('The city councilmen refused the demonstrators a permit because [they] advocated violence.')
# 'demonstrators'
```

See the [RoBERTA Winograd Schema Challenge (WSC) README](wsc/README.md) for more details on how to train this model.

#### Extract features aligned to words:

By default RoBERTa outputs one feature vector per BPE token. You can instead
realign the features to match [spaCy's word-level tokenization](https://spacy.io/usage/linguistic-features#tokenization)
with the `extract_features_aligned_to_words` method. This will compute a
weighted average of the BPE-level features for each word and expose them in
spaCy's `Token.vector` attribute:
```python
doc = roberta.extract_features_aligned_to_words('I said, "hello RoBERTa."')
assert len(doc) == 10
for tok in doc:
    print('{:10}{} (...)'.format(str(tok), tok.vector[:5]))
# <s>       tensor([-0.1316, -0.0386, -0.0832, -0.0477,  0.1943], grad_fn=<SliceBackward>) (...)
# I         tensor([ 0.0559,  0.1541, -0.4832,  0.0880,  0.0120], grad_fn=<SliceBackward>) (...)
# said      tensor([-0.1565, -0.0069, -0.8915,  0.0501, -0.0647], grad_fn=<SliceBackward>) (...)
# ,         tensor([-0.1318, -0.0387, -0.0834, -0.0477,  0.1944], grad_fn=<SliceBackward>) (...)
# "         tensor([-0.0486,  0.1818, -0.3946, -0.0553,  0.0981], grad_fn=<SliceBackward>) (...)
# hello     tensor([ 0.0079,  0.1799, -0.6204, -0.0777, -0.0923], grad_fn=<SliceBackward>) (...)
# RoBERTa   tensor([-0.2339, -0.1184, -0.7343, -0.0492,  0.5829], grad_fn=<SliceBackward>) (...)
# .         tensor([-0.1341, -0.1203, -0.1012, -0.0621,  0.1892], grad_fn=<SliceBackward>) (...)
# "         tensor([-0.1341, -0.1203, -0.1012, -0.0621,  0.1892], grad_fn=<SliceBackward>) (...)
# </s>      tensor([-0.0930, -0.0392, -0.0821,  0.0158,  0.0649], grad_fn=<SliceBackward>) (...)
```

#### Evaluating the `roberta.large.mnli` model:

Example python code snippet to evaluate accuracy on the MNLI `dev_matched` set.
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

- [Finetuning on GLUE](README.glue.md)
- [Finetuning on custom classification tasks (e.g., IMDB)](README.custom_classification.md)
- [Finetuning on Winograd Schema Challenge (WSC)](wsc/README.md)
- [Finetuning on Commonsense QA (CQA)](commonsense_qa/README.md)
- Finetuning on SQuAD: coming soon

## Pretraining using your own data

See the [tutorial for pretraining RoBERTa using your own data](README.pretraining.md).

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
