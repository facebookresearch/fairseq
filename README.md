# Introduction <img src="fairseq_logo.png" width="50"> 

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks. It provides reference implementations
of various sequence-to-sequence models, including:
- **Convolutional Neural Networks (CNN)**
  - [Dauphin et al. (2017): Language Modeling with Gated Convolutional Networks](examples/language_model/conv_lm/README.md)
  - [Gehring et al. (2017): Convolutional Sequence to Sequence Learning](examples/conv_seq2seq/README.md)
  - [Edunov et al. (2018): Classical Structured Prediction Losses for Sequence to Sequence Learning](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  - [Fan et al. (2018): Hierarchical Neural Story Generation](examples/stories/README.md)
- **LightConv and DynamicConv models**
  - **_New_** [Wu et al. (2019): Pay Less Attention with Lightweight and Dynamic Convolutions](examples/pay_less_attention_paper/README.md)
- **Long Short-Term Memory (LSTM) networks**
  - [Luong et al. (2015): Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
  - [Wiseman and Rush (2016): Sequence-to-Sequence Learning as Beam-Search Optimization](https://arxiv.org/abs/1606.02960)
- **Transformer (self-attention) networks**
  - [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - [Ott et al. (2018): Scaling Neural Machine Translation](examples/scaling_nmt/README.md)
  - [Edunov et al. (2018): Understanding Back-Translation at Scale](examples/backtranslation/README.md)
  - **_New_** [Baevski and Auli (2018): Adaptive Input Representations for Neural Language Modeling](examples/language_model/transformer_lm/README.md)
  - **_New_** [Shen et al. (2019): Mixture Models for Diverse Machine Translation: Tricks of the Trade](examples/translation_moe/README.md)

Fairseq features:
- multi-GPU (distributed) training on one machine or across multiple machines
- fast generation on both CPU and GPU with multiple search algorithms implemented:
  - beam search
  - Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  - sampling (unconstrained and top-k)
- large mini-batch training even on a single GPU via delayed updates
- fast half-precision floating point (FP16) training
- extensible: easily register new models, criterions, tasks, optimizers and learning rate schedulers

We also provide [pre-trained models](#pre-trained-models-and-examples) for several benchmark
translation and language modeling datasets.

![Model](fairseq.gif)

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6
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
