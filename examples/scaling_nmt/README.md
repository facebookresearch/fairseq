# Scaling Neural Machine Translation (Ott et al., 2018)

This page includes instructions for reproducing results from the paper [Scaling Neural Machine Translation (Ott et al., 2018)](https://arxiv.org/abs/1806.00187).

## Pre-trained models

Model | Description | Dataset | Download
---|---|---|---
`transformer.wmt14.en-fr` | Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
`transformer.wmt16.en-de` | Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)

## Training a new model on WMT'16 En-De

Please first download the [preprocessed WMT'16 En-De data provided by Google](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8).
Then:

1. Extract the WMT'16 En-De data:
```bash
TEXT=wmt16_en_de_bpe32k
mkdir $TEXT
tar -xzvf wmt16_en_de.tar.gz -C $TEXT
```

2. Preprocess the dataset with a joined dictionary:
```bash
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary
```

3. Train a model:
```bash
fairseq-train data-bin/wmt16_en_de_bpe32k \
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

## Citation

```bibtex
@inproceedings{ott2018scaling,
  title = {Scaling Neural Machine Translation},
  author = {Ott, Myle and Edunov, Sergey and Grangier, David and Auli, Michael},
  booktitle = {Proceedings of the Third Conference on Machine Translation (WMT)},
  year = 2018,
}
```
