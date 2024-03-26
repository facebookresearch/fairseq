# Scaling Neural Machine Translation (Ott et al., 2018)

This page includes instructions for reproducing results from the paper [Scaling Neural Machine Translation (Ott et al., 2018)](https://arxiv.org/abs/1806.00187).

## Pre-trained models

Model | Description | Dataset | Download
---|---|---|---
`transformer.wmt14.en-fr` | Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
`transformer.wmt16.en-de` | Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)

## Training a new model on WMT'16 En-De

First download the [preprocessed WMT'16 En-De data provided by Google](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8).

Then:

##### 1. Extract the WMT'16 En-De data
```bash
TEXT=wmt16_en_de_bpe32k
mkdir -p $TEXT
tar -xzvf wmt16_en_de.tar.gz -C $TEXT
```

##### 2. Preprocess the dataset with a joined dictionary
```bash
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
```

##### 3. Train a model
```bash
fairseq-train \
    data-bin/wmt16_en_de_bpe32k \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16
```

Note that the `--fp16` flag requires you have CUDA 9.1 or greater and a Volta GPU or newer.

***IMPORTANT:*** You will get better performance by training with big batches and
increasing the learning rate. If you want to train the above model with big batches
(assuming your machine has 8 GPUs):
- add `--update-freq 16` to simulate training on 8x16=128 GPUs
- increase the learning rate; 0.001 works well for big batches

##### 4. Evaluate

Now we can evaluate our trained model.

Note that the original [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
paper used a couple tricks to achieve better BLEU scores. We use these same tricks in
the Scaling NMT paper, so it's important to apply them when reproducing our results.

First, use the [average_checkpoints.py](/scripts/average_checkpoints.py) script to
average the last few checkpoints. Averaging the last 5-10 checkpoints is usually
good, but you may need to adjust this depending on how long you've trained:
```bash
python scripts/average_checkpoints \
    --inputs /path/to/checkpoints \
    --num-epoch-checkpoints 10 \
    --output checkpoint.avg10.pt
```

Next, generate translations using a beam width of 4 and length penalty of 0.6:
```bash
fairseq-generate \
    data-bin/wmt16_en_de_bpe32k \
    --path checkpoint.avg10.pt \
    --beam 4 --lenpen 0.6 --remove-bpe > gen.out
```

Finally, we apply the ["compound splitting" script](/scripts/compound_split_bleu.sh) to
add spaces around dashes. For example "Café-Liebhaber" would become three tokens:
"Café - Liebhaber". This typically results in larger BLEU scores, but it is not
appropriate to compare these inflated scores to work which does not include this trick.
This trick was used in the [original AIAYN code](https://github.com/tensorflow/tensor2tensor/blob/fc9335c0203685cbbfe2b30c92db4352d8f60779/tensor2tensor/utils/get_ende_bleu.sh),
so we used it in the Scaling NMT paper as well. That said, it's strongly advised to
report [sacrebleu](https://github.com/mjpost/sacrebleu) scores instead.

To compute "compound split" tokenized BLEU (not recommended!):
```bash
bash scripts/compound_split_bleu.sh gen.out
# BLEU4 = 29.29, 60.3/35.0/22.8/15.3 (BP=1.000, ratio=1.004, syslen=64763, reflen=64496)
```

To compute detokenized BLEU with sacrebleu (preferred):
```bash
bash scripts/sacrebleu.sh wmt14/full en de gen.out
# BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.3 = 28.6 59.3/34.3/22.1/14.9 (BP = 1.000 ratio = 1.016 hyp_len = 63666 ref_len = 62688)
```

## Citation

```bibtex
@inproceedings{ott2018scaling,
  title = {Scaling Neural Machine Translation},
  author = {Ott, Myle and Edunov, Sergey and Grangier, David and Auli, Michael},
  booktitle = {Proceedings of the Third Conference on Machine Translation (WMT)},
  year = 2018,
}
```
