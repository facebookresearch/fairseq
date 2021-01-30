# Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)

## Pre-trained models

Description | Parameters | Dataset | Model and Test set(s)
---|---:|---|---
Adaptive Inputs <br> ([Baevski and Auli, 2018](https://arxiv.org/abs/1809.10853)) | 1026M | [Google Billion Words](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2)
Adaptive Inputs <br> ([Baevski and Auli, 2018](https://arxiv.org/abs/1809.10853)) | 247M | [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2)

## Training an LM with adaptive inputs

First, see the general [language modeling README](README.md) for instructions on
preprocessing the WikiText-103 data.

Then use the following training command to train a model with adaptive inputs
using the `transformer_lm_wiki103` model architecture:
```bash
fairseq-train --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/transformer_wikitext-103 \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp
```

## Citation

```bibtex
@inproceedings{
    baevski2018adaptive,
    title={Adaptive Input Representations for Neural Language Modeling},
    author={Alexei Baevski and Michael Auli},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByxZX20qFQ},
}
```
