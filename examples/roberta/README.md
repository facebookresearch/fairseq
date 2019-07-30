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
```
>>> import torch
>>> roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
>>> roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Load RoBERTa (for PyTorch 1.0):
```
$ wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
$ tar -xzvf roberta.large.tar.gz

>>> from fairseq.models.roberta import RobertaModel
>>> roberta = RobertaModel.from_pretrained('/path/to/roberta.large')
>>> roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

##### Apply Byte-Pair Encoding (BPE) to input text:
```
>>> tokens = roberta.encode('Hello world!')
>>> tokens
tensor([    0, 31414,   232,   328,     2])
>>> roberta.decode(tokens)
'Hello world!'
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

##### Evaluating the `roberta.large.mnli` model

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

##### 1) Download the data from GLUE website (https://gluebenchmark.com/tasks) using following commands:
```
$ wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
$ python download_glue_data.py --data_dir glue_data --tasks all
```

##### 2) Preprocess GLUE task data:
```
$ ./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
```
`glue_task_name` is one of the following:
`{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}`
Use `ALL` for preprocessing all the glue tasks.

##### 3) Fine-tuning on GLUE task :
Example fine-tuning cmd for `RTE` task
```
TOTAL_NUM_UPDATES=2036  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=122      # 6 percent of the number of updates
LR=2e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size.

CUDA_VISIBLE_DEVICES=0 python train.py RTE-bin/ \
--restore-file <roberta_large_absolute_path> \
--max-positions 512 \
--max-sentences $MAX_SENTENCES \
--max-tokens 4400 \
--task sentence_prediction \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--init-token 0 --separator-token 2 \
--arch roberta_large \
--criterion sentence_prediction \
--num-classes $NUM_CLASSES \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
--clip-norm 0.0 \
--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
--max-epoch 10 \
--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
```

For each of the GLUE task, you will need to use following cmd-line arguments:

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-classes` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 2e-5 | 1e-5 | 1e-5 | 1e-5 | 2e-5
`--max-sentences` | 32 | 32 | 32 | 16 | 32 | 16 | 16 | 16
`--total-num-update` | 123873 | 33112 | 113272 | 2036 | 20935 | 2296 | 5336 | 3598
`--warmup-updates` | 7432 | 1986 | 28318 | 122 | 1256 | 137 | 320 | 214

For `STS-B` additionally use following cmd-line argument:
```
--regression-target
--best-checkpoint-metric loss
```
and remove `--maximize-best-checkpoint-metric`.

**Note:**

a) `--total-num-updates` is used by `--polynomial_decay` scheduler and is calculated for `--max-epoch=10` and `--max-sentences=16/32` depending on the task.

b) Above cmd-args and hyperparams are tested on one Nvidia `V100` GPU with `32gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq` and reduce `--max-sentences`.

c) All the settings in above table are suggested settings based on our hyperparam search within a fixed search space (for careful comparison across models). You might be able to find better metrics with wider hyperparam search.  

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
