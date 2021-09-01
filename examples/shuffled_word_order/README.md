# Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little

[https://arxiv.org/abs/2104.06644](https://arxiv.org/abs/2104.06644)

## Introduction

In this work, we pre-train [RoBERTa](../roberta) base on various word shuffled variants of BookWiki corpus (16GB). We observe that a word shuffled pre-trained model achieves surprisingly good scores on GLUE, PAWS and several parametric probing tasks. Please read our paper for more details on the experiments.

## Pre-trained models

| Model                                 | Description                                                                                        | Download                                                                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `roberta.base.orig`                   | RoBERTa (base) trained on natural corpus                                                           | [roberta.base.orig.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.orig.tar.gz)                                     |
| `roberta.base.shuffle.n1`             | RoBERTa (base) trained on n=1 gram sentence word shuffled data                                     | [roberta.base.shuffle.n1.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n1.tar.gz)                         |
| `roberta.base.shuffle.n2`             | RoBERTa (base) trained on n=2 gram sentence word shuffled data                                     | [roberta.base.shuffle.n2.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n2.tar.gz)                         |
| `roberta.base.shuffle.n3`             | RoBERTa (base) trained on n=3 gram sentence word shuffled data                                     | [roberta.base.shuffle.n3.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n3.tar.gz)                         |
| `roberta.base.shuffle.n4`             | RoBERTa (base) trained on n=4 gram sentence word shuffled data                                     | [roberta.base.shuffle.n4.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n4.tar.gz)                         |
| `roberta.base.shuffle.512`            | RoBERTa (base) trained on unigram 512 word block shuffled data                                     | [roberta.base.shuffle.512.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.512.tar.gz)                       |
| `roberta.base.shuffle.corpus`         | RoBERTa (base) trained on unigram corpus word shuffled data                                        | [roberta.base.shuffle.corpus.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.corpus.tar.gz)                 |
| `roberta.base.shuffle.corpus_uniform` | RoBERTa (base) trained on unigram corpus word shuffled data, where all words are uniformly sampled | [roberta.base.shuffle.corpus_uniform.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.corpus_uniform.tar.gz) |
| `roberta.base.nopos`                  | RoBERTa (base) without positional embeddings, trained on natural corpus                            | [roberta.base.nopos.tar.gz](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.nopos.tar.gz)                                   |

## Results

[GLUE (Wang et al, 2019)](https://gluebenchmark.com/) & [PAWS (Zhang et al, 2019)](https://github.com/google-research-datasets/paws) _(dev set, single model, single-task fine-tuning, median of 5 seeds)_

| name                                 |  CoLA |  MNLI |  MRPC |  PAWS |  QNLI |   QQP |   RTE | SST-2 |
| :----------------------------------- | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| `roberta.base.orig`                  |  61.4 | 86.11 | 89.19 | 94.46 | 92.53 | 91.26 | 74.64 | 93.92 |
| `roberta.base.shuffle.n1`            | 35.15 | 82.64 |    86 | 89.97 | 89.02 | 91.01 | 69.02 | 90.47 |
| `roberta.base.shuffle.n2`            | 54.37 | 83.43 | 86.24 | 93.46 | 90.44 | 91.36 | 70.83 | 91.79 |
| `roberta.base.shuffle.n3`            | 48.72 | 83.85 | 86.36 | 94.05 | 91.69 | 91.24 | 70.65 | 92.02 |
| `roberta.base.shuffle.n4`            | 58.64 | 83.77 | 86.98 | 94.32 | 91.69 |  91.4 | 70.83 | 92.48 |
| `roberta.base.shuffle.512`           | 12.76 | 77.52 | 79.61 | 84.77 | 85.19 |  90.2 | 56.52 | 86.34 |
| `roberta.base.shuffle.corpus`        |     0 |  71.9 | 70.52 | 58.52 | 71.11 | 85.52 | 53.99 | 83.35 |
| `roberta.base.shuffle.corpus_random` |  9.19 | 72.33 | 70.76 | 58.42 | 77.76 | 85.93 | 53.99 | 84.04 |
| `roberta.base.nopos`                 |     0 |  63.5 | 72.73 | 57.08 | 77.72 | 87.87 | 54.35 | 83.24 |

For more results on probing tasks, please refer to [our paper](https://arxiv.org/abs/2104.06644).

## Example Usage

Follow the same usage as in [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta) to load and test your models:

```python
# Download roberta.base.shuffle.n1 model
wget https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n1.tar.gz
tar -xzvf roberta.base.shuffle.n1.tar.gz

# Load the model in fairseq
from fairseq.models.roberta import RoBERTaModel
roberta = RoBERTaModel.from_pretrained('/path/to/roberta.base.shuffle.n1', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

**Note**: The model trained without positional embeddings (`roberta.base.nopos`) is a modified `RoBERTa` model, where the positional embeddings are not used. Thus, the typical `from_pretrained` method on fairseq version of RoBERTa will not be able to load the above model weights. To do so, construct a new `RoBERTaModel` object by setting the flag `use_positional_embeddings` to `False` (or [in the latest code](https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/model.py#L543), set `no_token_positional_embeddings` to `True`), and then load the individual weights.

## Fine-tuning Evaluation

We provide the trained fine-tuned models on MNLI here for each model above for quick evaluation (1 seed for each model). Please refer to [finetuning details](README.finetuning.md) for the parameters of these models. Follow [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta) instructions to evaluate these models.

| Model                                      | MNLI M Dev Accuracy | Link                                                                                                             |
| :----------------------------------------- | :------------------ | :--------------------------------------------------------------------------------------------------------------- |
| `roberta.base.orig.mnli`                   | 86.14               | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.orig.mnli.tar.gz)                   |
| `roberta.base.shuffle.n1.mnli`             | 82.55               | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n1.mnli.tar.gz)             |
| `roberta.base.shuffle.n2.mnli`             | 83.21               | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n2.mnli.tar.gz)             |
| `roberta.base.shuffle.n3.mnli`             | 83.89               | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n3.mnli.tar.gz)             |
| `roberta.base.shuffle.n4.mnli`             | 84.00               | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n4.mnli.tar.gz)             |
| `roberta.base.shuffle.512.mnli`            | 77.22               | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.512.mnli.tar.gz)            |
| `roberta.base.shuffle.corpus.mnli`         | 71.88               | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.corpus.mnli.tar.gz)         |
| `roberta.base.shuffle.corpus_uniform.mnli` | 72.46               | [Download](https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.corpus_uniform.mnli.tar.gz) |

## Citation

```bibtex
@misc{sinha2021masked,
      title={Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little},
      author={Koustuv Sinha and Robin Jia and Dieuwke Hupkes and Joelle Pineau and Adina Williams and Douwe Kiela},
      year={2021},
      eprint={2104.06644},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
