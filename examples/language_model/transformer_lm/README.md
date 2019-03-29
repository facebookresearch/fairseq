# Adaptive Input Representations for Neural Language Modeling (Baevski and Auli; 2018)

## Pre-trained models

Description | Parameters | Dataset | Model and Test set(s)
---|---:|---|---
Adaptive Inputs <br> ([Baevski and Auli, 2018](https://arxiv.org/abs/1809.10853)) | 1026M | [Google Billion Words](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.bz2)
Adaptive Inputs <br> ([Baevski and Auli, 2018](https://arxiv.org/abs/1809.10853)) | 247M | [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.bz2)

## Example usage

See the [language modeling README](../README.md) for instructions on reproducing results for WikiText-103
using the `transformer_lm_wiki103` model architecture.

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
