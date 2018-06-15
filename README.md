# Introduction

Fairseq(-py) is a sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks. It provides reference implementations of various sequence-to-sequence models, including:
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
- large mini-batch training (even on a single GPU) via delayed updates
- fast half-precision floating point (FP16) training

We also provide [pre-trained models](#pre-trained-models) for several benchmark translation datasets.

![Model](fairseq.gif)

# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6

Currently fairseq requires PyTorch version >= 0.4.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size` as command line
options to `nvidia-docker run`.

After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build
python setup.py develop
```

# Quick Start

The following command-line tools are provided:
* `python preprocess.py`: Data pre-processing: build vocabularies and binarize training data
* `python train.py`: Train a new model on one or multiple GPUs
* `python generate.py`: Translate pre-processed data with a trained model
* `python interactive.py`: Translate raw text with a trained model
* `python score.py`: BLEU scoring of generated translations against reference translations
* `python eval_lm.py`: Language model evaluation

## Evaluating Pre-trained Models
First, download a pre-trained model along with its vocabularies:
```
$ curl https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf -
```

This model uses a [Byte Pair Encoding (BPE) vocabulary](https://arxiv.org/abs/1508.07909), so we'll have to apply the encoding to the source text before it can be translated.
This can be done with the [apply_bpe.py](https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py) script using the `wmt14.en-fr.fconv-cuda/bpecodes` file.
`@@` is used as a continuation marker and the original text can be easily recovered with e.g. `sed s/@@ //g` or by passing the `--remove-bpe` flag to `generate.py`.
Prior to BPE, input text needs to be tokenized using `tokenizer.perl` from [mosesdecoder](https://github.com/moses-smt/mosesdecoder).

Let's use `python interactive.py` to generate translations interactively.
Here, we use a beam size of 5:
```
$ MODEL_DIR=wmt14.en-fr.fconv-py
$ python interactive.py \
 --path $MODEL_DIR/model.pt $MODEL_DIR \
 --beam 5
| loading model(s) from wmt14.en-fr.fconv-py/model.pt
| [en] dictionary: 44206 types
| [fr] dictionary: 44463 types
| Type the input sentence and press return:
> Why is it rare to discover new marine mam@@ mal species ?
O       Why is it rare to discover new marine mam@@ mal species ?
H       -0.06429661810398102    Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
A       0 1 3 3 5 6 6 8 8 8 7 11 12
```

This generation script produces four types of outputs: a line prefixed with *S* shows the supplied source sentence after applying the vocabulary; *O* is a copy of the original source sentence; *H* is the hypothesis along with an average log-likelihood; and *A* is the attention maxima for each word in the hypothesis, including the end-of-sentence marker which is omitted from the text.

Check [below](#pre-trained-models) for a full list of pre-trained models available.

## Training a New Model

The following tutorial is for machine translation.
For an example of how to use Fairseq for other tasks, such as [language modeling](examples/language_model/README.md), please see the `examples/` directory.

### Data Pre-processing

Fairseq contains example pre-processing scripts for several translation datasets: IWSLT 2014 (German-English), WMT 2014 (English-French) and WMT 2014 (English-German).
To pre-process and binarize the IWSLT dataset:
```
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..
$ TEXT=data/iwslt14.tokenized.de-en
$ python preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en
```
This will write binarized data that can be used for model training to `data-bin/iwslt14.tokenized.de-en`.

### Training
Use `python train.py` to train a new model.
Here a few example settings that work well for the IWSLT 2014 dataset:
```
$ mkdir -p checkpoints/fconv
$ CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --arch fconv_iwslt_de_en --save-dir checkpoints/fconv
```

By default, `python train.py` will use all available GPUs on your machine.
Use the [CUDA_VISIBLE_DEVICES](http://acceleware.com/blog/cudavisibledevices-masking-gpus) environment variable to select specific GPUs and/or to change the number of GPU devices that will be used.

Also note that the batch size is specified in terms of the maximum number of tokens per batch (`--max-tokens`).
You may need to use a smaller value depending on the available GPU memory on your system.

### Generation
Once your model is trained, you can generate translations using `python generate.py` **(for binarized data)** or `python interactive.py` **(for raw text)**:
```
$ python generate.py data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/fconv/checkpoint_best.pt \
  --batch-size 128 --beam 5
  | [de] dictionary: 35475 types
  | [en] dictionary: 24739 types
  | data-bin/iwslt14.tokenized.de-en test 6750 examples
  | model fconv
  | loaded checkpoint trainings/fconv/checkpoint_best.pt
  S-721   danke .
  T-721   thank you .
  ...
```

To generate translations with only a CPU, use the `--cpu` flag.
BPE continuation markers can be removed with the `--remove-bpe` flag.

# Pre-trained Models

We provide the following pre-trained fully convolutional sequence-to-sequence translation models:

* [wmt14.en-fr.fconv-py.tar.bz2](https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2): Pre-trained model for [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) including vocabularies
* [wmt14.en-de.fconv-py.tar.bz2](https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-de.fconv-py.tar.bz2): Pre-trained model for [WMT14 English-German](https://nlp.stanford.edu/projects/nmt) including vocabularies

We also provide pre-trained language models:
* [gbw_fconv_lm.tar.bz2](https://s3.amazonaws.com/fairseq-py/models/gbw_fconv_lm.tar.bz2): Pre-trained model for [Google Billion Words](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) including vocabularies
* [wiki103_fconv_lm.tar.bz2](https://s3.amazonaws.com/fairseq-py/models/wiki103_fconv_lm.tar.bz2): Pre-trained model for [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) including vocabularies

In addition, we provide pre-processed and binarized test sets for the models above:
* [wmt14.en-fr.newstest2014.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2): newstest2014 test set for WMT14 English-French
* [wmt14.en-fr.ntst1213.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.ntst1213.tar.bz2): newstest2012 and newstest2013 test sets for WMT14 English-French
* [wmt14.en-de.newstest2014.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-de.newstest2014.tar.bz2): newstest2014 test set for WMT14 English-German
* [wiki103_test_lm.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wiki103_test_lm.tar.bz2)
* [gbw_test_lm.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/gbw_test_lm.tar.bz2) 

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

# Large mini-batch training with delayed updates

The `--update-freq` option can be used to accumulate gradients from multiple mini-batches and delay updating,
creating a larger effective batch size.
Delayed updates can also improve training speed by reducing inter-GPU communication costs and by saving idle time caused by variance in workload across GPUs.
See [Ott et al. (2018)](https://arxiv.org/abs/1806.00187) for more details.

To train on a single GPU with an effective batch size that is equivalent to training on 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0 python train.py --update-freq 8 (...)
```

# Training with half precision floating point (FP16)

> Note: FP16 training requires a Volta GPU and CUDA 9.1 or greater

Recent GPUs enable efficient half precision floating point computation, e.g., using [Nvidia Tensor Cores](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html).

Fairseq supports FP16 training with the `--fp16` flag:
```
python train.py --fp16 (...)
```

# Distributed training

Distributed training in fairseq is implemented on top of [torch.distributed](http://pytorch.org/docs/master/distributed.html).
Training begins by launching one worker process per GPU.
These workers discover each other via a unique host and port (required) that can be used to establish an initial connection.
Additionally, each worker has a rank, that is a unique number from 0 to n-1 where n is the total number of GPUs.

If you run on a cluster managed by [SLURM](https://slurm.schedmd.com/) you can train a large English-French model on the WMT 2014 dataset on 16 nodes with 8 GPUs each (in total 128 GPUs) using this command:

```
$ DATA=...   # path to the preprocessed dataset, must be visible from all nodes
$ PORT=9218  # any available TCP port that can be used by the trainer to establish initial connection
$ sbatch --job-name fairseq-py --gres gpu:8 --cpus-per-task 10 \
    --nodes 16 --ntasks-per-node 8 \
    --wrap 'srun --output train.log.node%t --error train.stderr.node%t.%j \
    python train.py $DATA \
    --distributed-world-size 128 \
    --distributed-port $PORT \
    --force-anneal 50 --lr-scheduler fixed --max-epoch 55 \
    --arch fconv_wmt_en_fr --optimizer nag --lr 0.1,4 --max-tokens 3000 \
    --clip-norm 0.1 --dropout 0.1 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --wd 0.0001'
```

Alternatively you can manually start one process per GPU:
```
$ DATA=...  # path to the preprocessed dataset, must be visible from all nodes
$ HOST_PORT=master.devserver.com:9218  # one of the hosts used by the job
$ RANK=...  # the rank of this process, from 0 to 127 in case of 128 GPUs
$ python train.py $DATA \
    --distributed-world-size 128 \
    --distributed-init-method 'tcp://$HOST_PORT' \
    --distributed-rank $RANK \
    --force-anneal 50 --lr-scheduler fixed --max-epoch 55 \
    --arch fconv_wmt_en_fr --optimizer nag --lr 0.1,4 --max-tokens 3000 \
    --clip-norm 0.1 --dropout 0.1 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --wd 0.0001
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
This is a PyTorch version of [fairseq](https://github.com/facebookresearch/fairseq), a sequence-to-sequence learning toolkit from Facebook AI Research. The original authors of this reimplementation are (in no particular order) Sergey Edunov, Myle Ott, and Sam Gross.
