# Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)

## Pre-trained models

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)) | [Google Billion Words](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/lm/gbw_fconv_lm.tar.bz2) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/gbw_test_lm.tar.bz2)
Convolutional <br> ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)) | [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wiki103_fconv_lm.tar.bz2) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wiki103_test_lm.tar.bz2)

## Example usage

See the [language modeling README](../language_model/README.md) for instructions on reproducing results for WikiText-103
using the `fconv_lm_dauphin_wikitext103` model architecture.

## Citation

```bibtex
@inproceedings{dauphin2017language,
  title={Language Modeling with Gated Convolutional Networks},
  author={Dauphin, Yann N and Fan, Angela and Auli, Michael and Grangier, David},
  booktitle={Proceedings of the 34th International Conference on Machine Learning-Volume 70},
  pages={933--941},
  year={2017},
  organization={JMLR}
}
```
