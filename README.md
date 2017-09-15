# Introduction
FAIR Sequence-to-Sequence Toolkit (PyTorch)

This is a PyTorch version of [fairseq](https://github.com/facebookresearch/fairseq), a sequence-to-sequence learning toolkit from Facebook AI Research. The original authors of this reimplementation are (in no particular order) Sergey Edunov, Myle Ott, and Sam Gross. The toolkit implements the fully convolutional model described in [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122). The toolkit features multi-GPU training on a single machine as well as fast beam search generation on both CPU and GPU. We provide pre-trained models for English to French and English to German translation.

![Model](fairseq.gif)

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

# Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

Currently fairseq-py requires PyTorch from the GitHub repository. There are multiple ways of installing it.
We suggest using [Miniconda3](https://conda.io/miniconda.html) and the following instructions.

* Install Miniconda3 from https://conda.io/miniconda.html create and activate python 3 environment.

```
conda install gcc numpy cudnn nccl
conda install magma-cuda80 -c soumith
pip install cmake
pip install cffi

git clone https://github.com/pytorch/pytorch.git
cd pytorch
git reset --hard a03e5cb40938b6b3f3e6dbddf9cff8afdff72d1b
git submodule update --init
pip install -r requirements.txt

NO_DISTRIBUTED=1 python setup.py install
```


Install fairseq by cloning the GitHub repository and by running

```
pip install -r requirements.txt
python setup.py build
python setup.py develop
```

The following command-line tools are available:
* `python preprocess.py`: Data pre-processing: build vocabularies and binarize training data
* `python train.py`: Train a new model on one or multiple GPUs
* `python generate.py`: Translate pre-processed data with a trained model
* `python generate.py -i`: Translate raw text with a trained model
* `python score.py`: BLEU scoring of generated translations against reference translations


# Quick Start

## Evaluating Pre-trained Models [TO BE ADAPTED]
First, download a pre-trained model along with its vocabularies:
```
$ curl https://s3.amazonaws.com/fairseq-py/models/wmt14.en-fr.fconv-py.tar.bz2 | tar xvjf -
```

This model uses a [Byte Pair Encoding (BPE) vocabulary](https://arxiv.org/abs/1508.07909), so we'll have to apply the encoding to the source text before it can be translated.
This can be done with the [apply_bpe.py](https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py) script using the `wmt14.en-fr.fconv-cuda/bpecodes` file.
`@@` is used as a continuation marker and the original text can be easily recovered with e.g. `sed s/@@ //g` or by passing the `--remove-bpe` flag to `generate.py`.
Prior to BPE, input text needs to be tokenized using `tokenizer.perl` from [mosesdecoder](https://github.com/moses-smt/mosesdecoder).

Let's use `python generate.py -i` to generate translations.
Here, we use a beam size of 5:
```
$ MODEL_DIR=wmt14.en-fr.fconv-py
$ python generate.py -i \
 --path $MODEL_DIR/model.pt $MODEL_DIR \
 --beam 5
| [en] dictionary: 44206 types
| [fr] dictionary: 44463 types
| model fconv_wmt_en_fr
| loaded checkpoint /private/home/edunov/wmt14.en-fr.fconv-py/model.pt (epoch 37)
> Why is it rare to discover new marine mam@@ mal species ?
S       Why is it rare to discover new marine mam@@ mal species ?
O       Why is it rare to discover new marine mam@@ mal species ?
H       -0.08662842959165573    Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
A       0 1 3 3 5 6 6 10 8 8 8 11 12
```

This generation script produces four types of outputs: a line prefixed with *S* shows the supplied source sentence after applying the vocabulary; *O* is a copy of the original source sentence; *H* is the hypothesis along with an average log-likelihood; and *A* is the attention maxima for each word in the hypothesis, including the end-of-sentence marker which is omitted from the text.

Check [below](#pre-trained-models) for a full list of pre-trained models available.


## Training a New Model

### Data Pre-processing
The fairseq source distribution contains an example pre-processing script for
the IWSLT 2014 German-English corpus.
Pre-process and binarize the data as follows:
```
$ cd data/
$ bash prepare-iwslt14.sh
$ cd ..
$ TEXT=data/iwslt14.tokenized.de-en
$ python preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --thresholdtgt 3 --thresholdsrc 3 --destdir data-bin/iwslt14.tokenized.de-en
```
This will write binarized data that can be used for model training to `data-bin/iwslt14.tokenized.de-en`.

### Training
Use `python train.py` to train a new model.
Here a few example settings that work well for the IWSLT 2014 dataset:
```
$ mkdir -p trainings/fconv
$ CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --encoder-layers "[(256, 3)] * 4" --decoder-layers "[(256, 3)] * 3" \
  --encoder-embed-dim 256 --decoder-embed-dim 256 --save-dir trainings/fconv
```

By default, `python train.py` will use all available GPUs on your machine.
Use the [CUDA_VISIBLE_DEVICES](http://acceleware.com/blog/cudavisibledevices-masking-gpus) environment variable to select specific GPUs and/or to change the number of GPU devices that will be used.

Also note that the batch size is specified in terms of the maximum number of tokens per batch (`--max-tokens`).
You may need to use a smaller value depending on the available GPU memory on your system.

### Generation
Once your model is trained, you can generate translations using `python generate.py` **(for binarized data)** or `python generate.py -i` **(for raw text)**:
```
$ python generate.py data-bin/iwslt14.tokenized.de-en \
  --path trainings/fconv/checkpoint_best.pt \
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

We provide the following pre-trained fully convolutional sequence-to-sequence models:

* [wmt14.en-fr.fconv-py.tar.bz2](https://s3.amazonaws.com/faiseq-py/models/wmt14.en-fr.fconv-py.tar.bz2): Pre-trained model for [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) including vocabularies
* [wmt14.en-de.fconv-py.tar.bz2](https://s3.amazonaws.com/faiseq-py/models/wmt14.en-de.fconv-py.tar.bz2): Pre-trained model for [WMT14 English-German](https://nlp.stanford.edu/projects/nmt) including vocabularies

In addition, we provide pre-processed and binarized test sets for the models above:
* [wmt14.en-fr.newstest2014.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.en-fr.newstest2014.tar.bz2): newstest2014 test set for WMT14 English-French
* [wmt14.en-fr.ntst1213.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.en-fr.ntst1213.tar.bz2): newstest2012 and newstest2013 test sets for WMT14 English-French
* [wmt14.en-de.newstest2014.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.en-de.newstest2014.tar.bz2): newstest2014 test set for WMT14 English-German

Generation with the binarized test sets can be run in batch mode as follows, e.g. for English-French on a GTX-1080ti:
```
$ curl https://s3.amazonaws.com/faiseq-py/models/wmt14.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://s3.amazonaws.com/fairseq-py/data/wmt14.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ python generate.py data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (95451 tokens) in 136.3s (700.49 tokens/s)
| Timings: setup 0.1s (0.1%), encoder 1.9s (1.4%), decoder 108.9s (79.9%), search_results 0.0s (0.0%), search_prune 12.5s (9.2%)
| BLEU4 = 43.43, 68.2/49.2/37.4/28.8 (BP=0.996, ratio=1.004, sys_len=92087, ref_len=92448)

# Word-level BLEU scoring:
$ python score.py --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
TODO: update scores
BLEU4 = 40.55, 67.6/46.5/34.0/25.3 (BP=1.000, ratio=0.998, sys_len=81369, ref_len=81194)
```

# Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License
fairseq is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.
