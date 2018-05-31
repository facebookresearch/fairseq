Sample data processing scripts for the FAIR Sequence-to-Sequence Toolkit

These scripts provide an example of pre-processing data for the NMT task.

# prepare-iwslt14.sh

Provides an example of pre-processing for IWSLT'14 German to English translation task: ["Report on the 11th IWSLT evaluation campaign" by Cettolo et al.](http://workshop2014.iwslt.org/downloads/proceeding.pdf)

Example usage:
```
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=data/iwslt14.tokenized.de-en
$ python preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en

# Train the model (better for a single GPU setup):
$ mkdir -p checkpoints/fconv
$ CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 200 \
  --arch fconv_iwslt_de_en --save-dir checkpoints/fconv

# Generate:
$ python generate.py data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/fconv/checkpoint_best.pt \
  --batch-size 128 --beam 5 --remove-bpe

```


# prepare-wmt14en2de.sh

Provides an example of pre-processing for the WMT'14 English to German translation task. By default it will produce a dataset that was modeled after ["Attention Is All You Need" by Vaswani et al.](https://arxiv.org/abs/1706.03762) that includes news-commentary-v12 data.

To use only data available in WMT'14 or to replicate results obtained in the original paper ["Convolutional Sequence to Sequence Learning" by Gehring et al.](https://arxiv.org/abs/1705.03122) run it with --icml17 instead:

```
$ bash prepare-wmt14en2de.sh --icml17
```

Example usage:

```
$ cd examples/translation/
$ bash prepare-wmt14en2de.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=data/wmt14_en_de
$ python preprocess.py --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1500 instead
$ mkdir -p checkpoints/fconv_wmt_en_de
$ python train.py data-bin/wmt14_en_de \
  --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --arch fconv_wmt_en_de --save-dir checkpoints/fconv_wmt_en_de

# Generate:
$ python generate.py data-bin/wmt14_en_de \
  --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt --beam 5 --remove-bpe

```

# prepare-wmt14en2fr.sh

Provides an example of pre-processing for the WMT'14 English to French translation task.

Example usage:

```
$ cd examples/translation/
$ bash prepare-wmt14en2fr.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=data/wmt14_en_fr
$ python preprocess.py --source-lang en --target-lang fr \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1000 instead
$ mkdir -p checkpoints/fconv_wmt_en_fr
$ python train.py data-bin/wmt14_en_fr \
  --lr 0.5 --clip-norm 0.1 --dropout 0.1 --max-tokens 3000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --arch fconv_wmt_en_fr --save-dir checkpoints/fconv_wmt_en_fr

# Generate:
$ python generate.py data-bin/fconv_wmt_en_fr \
  --path checkpoints/fconv_wmt_en_fr/checkpoint_best.pt --beam 5 --remove-bpe

```

