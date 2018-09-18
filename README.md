# Introduction

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks. It provides reference implementations
of various sequence-to-sequence models, including:
- **Convolutional Neural Networks (CNN)**
  - [Dauphin et al. (2017): Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
  - [Gehring et al. (2017): Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
  - **_New_** [Edunov et al. (2018): Classical Structured Prediction Losses for Sequence to Sequence Learning](https://arxiv.org/abs/1711.04956)
  - **_New_** [Fan et al. (2018): Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)
- **Long Short-Term Memory (LSTM) networks**
  - [Luong et al. (2015): Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
  - [Wiseman and Rush (2016): Sequence-to-Sequence Learning as Beam-Search Optimization](https://arxiv.org/abs/1606.02960)
- **Transformer (self-attention) networks**
  - [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - **_New_** [Ott et al. (2018): Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187)

Fairseq features:
- multi-GPU (distributed) training on one machine or across multiple machines
- fast beam search generation on both CPU and GPU
- large mini-batch training even on a single GPU via delayed updates
- fast half-precision floating point (FP16) training
- extensible: easily register new models, criterions, and tasks

We also provide [pre-trained models](#pre-trained-models) for several benchmark
translation and language modeling datasets.

![Model](fairseq.gif)

# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6

Currently fairseq requires PyTorch version >= 0.4.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.

After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build develop
```

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Pre-trained Models

We provide the following pre-trained models and pre-processed, binarized test sets:

### Translation

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt14.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt14.en-de.newstest2014.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt17.v2.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt17.v2.en-de.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt16.en-de.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)

### Language models

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)) | [Google Billion Words](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/gbw_fconv_lm.tar.bz2) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/gbw_test_lm.tar.bz2)
Convolutional <br> ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)) | [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wiki103_fconv_lm.tar.bz2) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wiki103_test_lm.tar.bz2)

### Stories

Description | Dataset | Model | Test set(s)
---|---|---|---
Stories with Convolutional Model <br> ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)) | [WritingPrompts](https://arxiv.org/abs/1805.04833) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/stories_checkpoint.tar.bz2) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/stories_test.tar.bz2)


### Usage

Generation with the binarized test sets can be run in batch mode as follows, e.g. for WMT 2014 English-French on a GTX-1080ti:
```
$ curl https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ python generate.py data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
| Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Scoring with score.py:
$ grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
$ python score.py --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

# Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# Citation

If you use the code in your paper, then please cite it as:

```
@inproceedings{gehring2017convs2s,
  author    = {Gehring, Jonas, and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  title     = "{Convolutional Sequence to Sequence Learning}",
  booktitle = {Proc. of ICML},
  year      = 2017,
}
```

# License
fairseq(-py) is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.

# Credits
This is a PyTorch version of
[fairseq](https://github.com/facebookresearch/fairseq), a sequence-to-sequence
learning toolkit from Facebook AI Research. The original authors of this
reimplementation are (in no particular order) Sergey Edunov, Myle Ott, and Sam
Gross.
