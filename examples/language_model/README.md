Sample data processing scripts for the FAIR Sequence-to-Sequence Toolkit

These scripts provide an example of pre-processing data for the Language Modeling task.

# prepare-wikitext-103.sh

Provides an example of pre-processing for [WikiText-103 language modeling task](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset):

Example usage:
```
$ cd examples/language_model/
$ bash prepare-wikitext-103.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/language_model/wikitext-103

$ python preprocess.py --only-source \
  --trainpref $TEXT/wiki.train.tokens --validpref $TEXT/wiki.valid.tokens --testpref $TEXT/wiki.test.tokens \ 
  --destdir data-bin/wikitext-103

# Train the model:
# If it runs out of memory, try to reduce max-tokens and max-target-positions
$ mkdir -p checkpoints/wikitext-103
$ python train.py data-bin/wikitext-103 --save-dir /checkpoints/wikitext-103 \ 
  --max-epoch 35 --arch fconv_lm --optimizer nag --lr 1.0 --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
  --decoder-layers '[(850, 6)] * 3 + [(850,1)] + [(850,5)] * 4 + [(850,1)] + [(850,4)] * 3 + [(1024,4)] + [(2048, 4)]' \ 
  --decoder-embed-dim 280 --clip-norm 0.1 --dropout 0.2 --weight-decay 5e-06 --criterion adaptive_loss \
  --adaptive-softmax-cutoff 10000,20000,200000 --max-tokens 1024 --max-target-positions 1024

# Evaluate:
$ python eval_lm.py data-bin/wikitext-103 --path 'checkpoints/wiki103/checkpoint_best.pt'

```
