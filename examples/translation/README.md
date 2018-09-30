# Example usage for Neural Machine Translation

These scripts provide an example of pre-processing data for the NMT task
and instructions for how to replicate the results from the paper [Scaling Neural Machine Translation (Ott et al., 2018)](https://arxiv.org/abs/1806.00187).

## Preprocessing

### prepare-iwslt14.sh

Provides an example of pre-processing for IWSLT'14 German to English translation task: ["Report on the 11th IWSLT evaluation campaign" by Cettolo et al.](http://workshop2014.iwslt.org/downloads/proceeding.pdf)

Example usage:
```
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/iwslt14.tokenized.de-en
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

To train transformer model on IWSLT'14 German to English:
```
# Preparation steps are the same as for fconv model.

# Train the model (better for a single GPU setup):
$ mkdir -p checkpoints/transformer
$ CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en \
  -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s de -t en \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/transformer

# Average 10 latest checkpoints:
$ python scripts/average_checkpoints.py --inputs checkpoints/transformer \
   --num-epoch-checkpoints 10 --output checkpoints/transformer/model.pt

# Generate:
$ python generate.py data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/transformer/model.pt \
  --batch-size 128 --beam 5 --remove-bpe

```


### prepare-wmt14en2de.sh

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
$ TEXT=examples/translation/wmt14_en_de
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

### prepare-wmt14en2fr.sh

Provides an example of pre-processing for the WMT'14 English to French translation task.

Example usage:

```
$ cd examples/translation/
$ bash prepare-wmt14en2fr.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/wmt14_en_fr
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

## Replicating results from "Scaling Neural Machine Translation"

To replicate results from the paper [Scaling Neural Machine Translation (Ott et al., 2018)](https://arxiv.org/abs/1806.00187),
please first download the [preprocessed WMT'16 En-De data provided by Google](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8).

1. Extract the WMT'16 En-De data:
```
$ TEXT=wmt16_en_de_bpe32k
$ mkdir $TEXT
$ tar -xzvf wmt16_en_de.tar.gz -C $TEXT
```
2. Preprocess the dataset with a joined dictionary:
```
$ python preprocess.py --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir data-bin/wmt16_en_de_bpe32k \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary
```
3. Train a model:
```
$ python train.py data-bin/wmt16_en_de_bpe32k \
  --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0005 --min-lr 1e-09 \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 3584 \
  --fp16
```

Note that the `--fp16` flag requires you have CUDA 9.1 or greater and a Volta GPU.

If you want to train the above model with big batches (assuming your machine has 8 GPUs):
- add `--update-freq 16` to simulate training on 8*16=128 GPUs
- increase the learning rate; 0.001 works well for big batches
