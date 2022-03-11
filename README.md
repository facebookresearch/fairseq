<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p>

--------------------------------------------------------------------------------

# Classical Structured Prediction Losses for Sequence to Sequence Learning

We provide the code to reproduce results obtained in the paper: https://arxiv.org/abs/1711.04956.
The code is based on fairseq-py, please refer to the [fairseq-py README](https://github.com/pytorch/fairseq) for general usage instructions.

This branch is using an older version of pytorch v0.3.1, we recommend installing it from the source: https://github.com/pytorch/pytorch/tree/v0.3.1

For IWSLT dataset, first train a baseline model (every command here assumes 8 GPU setup):
```
python ./train.py $DATA -a fconv_iwslt_de_en  --lr 0.25 \
    --clip-norm 0.1 --dropout 0.3 --max-tokens 1000 \
    --label-smoothing 0.1 --force-anneal 200 --save-dir checkpoints_tls
```
Please, refer to "Data Pre-processing" section of this readme to get the dataset.

Then, to fine-tune the trained model with sequence level training, do one of the following commands.

SeqNLL:
```
mkdir checkpoints_seqnll
cp checkpoints_tls/checkpoint200.pt checkpoints_seqnll/checkpoint_last.pt
python train.py $DATA -a fconv_iwslt_de_en --clip-norm 0.1 \
    --momentum 0.9 --lr 0.25  \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 1000 \
    --seq-max-len-a 1.5 --seq-max-len-b 5 --seq-criterion SequenceCrossEntropyCriterion \
    --seq-combined-loss-alpha 0.4 --force-anneal 220 --seq-beam 16 --save-dir checkpoints_seqnll
```

Risk:
```
mkdir checkpoints_risk
cp checkpoints_tls/checkpoint200.pt checkpoints_risk/checkpoint_last.pt
python train.py $DATA -a fconv_iwslt_de_en --clip-norm 0.1 \
    --momentum 0.9 --lr 0.25  \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 1000 \
    --seq-max-len-a 1.5 --seq-max-len-b 5 --seq-criterion SequenceRiskCriterion \
    --seq-combined-loss-alpha 0.5 --force-anneal 220 --seq-beam 16 --save-dir checkpoints_risk
```

Max-margin:
```
mkdir checkpoints_maxmargin
cp checkpoints_tls/checkpoint200.pt checkpoints_maxmargin/checkpoint_last.pt
python train.py $DATA -a fconv_iwslt_de_en --clip-norm 0.1 \
    --momentum 0.9 --lr 0.25  \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 1000 \
    --seq-max-len-a 1.5 --seq-max-len-b 5 --seq-criterion SequenceMaxMarginCriterion \
    --seq-combined-loss-alpha 0.4 --force-anneal 220 --seq-beam 16 --save-dir checkpoints_maxmargin
```

Multi-Margin:
```
mkdir checkpoints_multimargin
cp checkpoints_tls/checkpoint200.pt checkpoints_multimargin/checkpoint_last.pt
python train.py $DATA -a fconv_iwslt_de_en --clip-norm 0.1 \
    --momentum 0.9 --lr 0.25  \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 1000 \
    --seq-max-len-a 1.5 --seq-max-len-b 5 --seq-criterion SequenceMultiMarginCriterion \
    --seq-combined-loss-alpha 0.4 --force-anneal 220 --seq-beam 16 --save-dir checkpoints_multimargin
```

Softmax-Margin:
```
mkdir checkpoints_softmaxmargin
cp checkpoints_tls/checkpoint200.pt checkpoints_softmaxmargin/checkpoint_last.pt
python train.py $DATA -a fconv_iwslt_de_en --clip-norm 0.1 \
    --momentum 0.9 --lr 0.25  \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 1000 \
    --seq-max-len-a 1.5 --seq-max-len-b 5 --seq-criterion SequenceSoftMaxMarginCriterion \
    --seq-combined-loss-alpha 0.4 --force-anneal 220 --seq-beam 16 --save-dir checkpoints_softmaxmargin
```

Then, you can evaluate models normally.

# License
fairseq-py is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.

