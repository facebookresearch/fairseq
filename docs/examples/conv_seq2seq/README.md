# Convolutional Sequence to Sequence Learning (Gehring et al., 2017)

## Pre-trained models

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-de.newstest2014.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.v2.en-de.newstest2014.tar.bz2)

## Example usage

See the [translation README](../translation/README.md) for instructions on reproducing results for WMT'14 En-De and
WMT'14 En-Fr using the `fconv_wmt_en_de` and `fconv_wmt_en_fr` model architectures.

## Citation

```bibtex
@inproceedings{gehring2017convs2s,
  title = {Convolutional Sequence to Sequence Learning},
  author = {Gehring, Jonas, and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  booktitle = {Proc. of ICML},
  year = 2017,
}
```
