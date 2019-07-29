# <img src="fairseq_logo.png" width="30"> Introduction

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.

### What's New:

- July 2019: [RoBERTa models and code release](examples/roberta/README.md)
- June 2019: [wav2vec models and code release](examples/wav2vec/README.md)
- April 2019: [fairseq demo paper @ NAACL 2019](https://arxiv.org/abs/1904.01038)

### Features:

Fairseq provides reference implementations of various sequence-to-sequence models, including:
- **Convolutional Neural Networks (CNN)**
  - [Dauphin et al. (2017): Language Modeling with Gated Convolutional Networks](examples/language_model/conv_lm/README.md)
  - [Gehring et al. (2017): Convolutional Sequence to Sequence Learning](examples/conv_seq2seq/README.md)
  - [Edunov et al. (2018): Classical Structured Prediction Losses for Sequence to Sequence Learning](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  - [Fan et al. (2018): Hierarchical Neural Story Generation](examples/stories/README.md)
  - **_New_** [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
- **LightConv and DynamicConv models**
  - [Wu et al. (2019): Pay Less Attention with Lightweight and Dynamic Convolutions](examples/pay_less_attention_paper/README.md)
- **Long Short-Term Memory (LSTM) networks**
  - Luong et al. (2015): Effective Approaches to Attention-based Neural Machine Translation
- **Transformer (self-attention) networks**
  - Vaswani et al. (2017): Attention Is All You Need
  - [Ott et al. (2018): Scaling Neural Machine Translation](examples/scaling_nmt/README.md)
  - [Edunov et al. (2018): Understanding Back-Translation at Scale](examples/backtranslation/README.md)
  - [Baevski and Auli (2018): Adaptive Input Representations for Neural Language Modeling](examples/language_model/transformer_lm/README.md)
  - [Shen et al. (2019): Mixture Models for Diverse Machine Translation: Tricks of the Trade](examples/translation_moe/README.md)
  - **_New_** [Liu et al. (2019): RoBERTa: A Robustly Optimized BERT Pretraining Approach](examples/roberta/README.md)

**Additionally:**
- multi-GPU (distributed) training on one machine or across multiple machines
- fast generation on both CPU and GPU with multiple search algorithms implemented:
  - beam search
  - Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  - sampling (unconstrained, top-k and top-p/nucleus)
- large mini-batch training even on a single GPU via delayed updates
- mixed precision training (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
- extensible: easily register new models, criterions, tasks, optimizers and learning rate schedulers

We also provide [pre-trained models](#pre-trained-models-and-examples) for several benchmark
translation and language modeling datasets.

![Model](fairseq.gif)

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.5
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

Please follow the instructions here to install PyTorch: https://github.com/pytorch/pytorch#installation.

If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.

After PyTorch is installed, you can install fairseq with `pip`:
```
pip install fairseq
```

**Installing from source**

To install fairseq from source and develop locally:
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable .
```

**Improved training speed**

Training speed can be further improved by installing NVIDIA's
[apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` option.
fairseq will automatically switch to the faster modules provided by apex.

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Pre-trained models and examples

We provide pre-trained models and pre-processed, binarized test sets for several tasks listed below,
as well as example training and evaluation commands.

- [Translation](examples/translation/README.md): convolutional and transformer models are available
- [Language Modeling](examples/language_model/README.md): convolutional models are available

We also have more detailed READMEs to reproduce results from specific papers:
- [Liu et al. (2019): RoBERTa: A Robustly Optimized BERT Pretraining Approach](examples/roberta/README.md)
- [Schneider et al. (2019): wav2vec: Unsupervised Pre-training for Speech Recognition](examples/wav2vec/README.md)
- [Shen et al. (2019) Mixture Models for Diverse Machine Translation: Tricks of the Trade](examples/translation_moe/README.md)
- [Wu et al. (2019): Pay Less Attention with Lightweight and Dynamic Convolutions](examples/pay_less_attention_paper/README.md)
- [Edunov et al. (2018): Understanding Back-Translation at Scale](examples/backtranslation/README.md)
- [Edunov et al. (2018): Classical Structured Prediction Losses for Sequence to Sequence Learning](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
- [Fan et al. (2018): Hierarchical Neural Story Generation](examples/stories/README.md)
- [Ott et al. (2018): Scaling Neural Machine Translation](examples/scaling_nmt/README.md)
- [Gehring et al. (2017): Convolutional Sequence to Sequence Learning](examples/conv_seq2seq/README.md)
- [Dauphin et al. (2017): Language Modeling with Gated Convolutional Networks](examples/language_model/conv_lm/README.md)

# Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License
fairseq(-py) is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.

# Citation

Please cite as:

```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
