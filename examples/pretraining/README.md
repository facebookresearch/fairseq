# Pre-trained Language Model Representations for Language Generation

This page includes pre-trained models from the paper [Pre-trained Language Model Representations for Language Generation (Edunov et al., 2019)](https://arxiv.org/abs/1903.09722).

## Citation:

```bibtex
@article{edunov2019pre,
    author = {{Edunov}, S. and {Baevski}, A. and {Auli}, M.},
    title = "{Pre-trained Language Model Representations for Language Generation}",
    year = 2019,
    url = {https://arxiv.org/pdf/1903.09722.pdf}
}
```

## Pre-trained models

Description | Dataset | Model
---|---|---
English to German Transformer with pre-trained encoder | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/pretraining/en2de_mt.tar.gz)
Abstractive summarization, transformer | CNN-DailyMail dataset | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/pretraining/cnn_dailymail.tar.gz)


# Pre-training for Machine Translation

First step is to train a language model.
${LM_DATA} has to point to a folder with processed monolingual dataset, e.g. newscrawl. It is essential
to process the dataset with the same BPE and dictionary as the source side of machine translation data.

```
# Requires 32 GPUs:

$ python train.py ${LM_DATA} -a bi_transformer_lm_big --clip-norm 0.1 \
   --lr 0.0001 --dropout 0.1 --max-tokens 2480 --no-progress-bar --log-interval 1 \
   --criterion cross_entropy --fp16  --optimizer nag --lr-scheduler cosine \
   --warmup-init-lr 1e-07 --warmup-updates 16000 --min-lr 1e-09 \
   --max-update 984000 --task language_modeling --max-lr 1.0 --lr-period-updates 968000 \
   --lr-shrink 1.0 --decoder-layers 12 --attention-dropout 0.1 --decoder-embed-dim 1024 \
   --ddp-backend legacy --sample-break-mode eos --skip-invalid-size-inputs-valid-test \
   --relu-dropout 0.05 --save-interval-updates 200000 --keep-interval-updates 10 \
   --distributed-port 12597 --distributed-world-size 32 --save-dir ${LM_CHECKPOINT_PATH}

```

Next, to train a machine-translation model.
${MT_DATA} is a processed machine translation dataset, e.g. WMT En2De.

```
$ python train.py ${MT_DATA} -a transformer_wmt_en_de_big --no-enc-token-positional-embeddings \
   --elmo-affine --elmo-softmax --clip-norm 0 --fp16 --optimizer adam --lr 0.0007 \
   --label-smoothing 0.1 --ddp-backend no_c10d --dropout 0.3 --elmo-dropout 0.2 \
   --distributed-port 12597 --distributed-world-size 128 --max-tokens 3584 --no-progress-bar \
   --log-interval 100 --seed 1 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0 \
   --criterion label_smoothed_cross_entropy --max-update 30000 --share-decoder-input-output-embed \
   --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' \
   --encoder-embed-path elmo:${LM_CHECKPOINT_PATH}/checkpoint_best.pt
```

## To generate using the pre-trained model

```
curl https://dl.fbaipublicfiles.com/fairseq/models/pretraining/en2de_mt.tar.gz | tar xvzf -
cd en2de_mt

MOSES=... # path to moses tokenizer
BPEROOT=... # path to subword-nmt
FAIRSEQ=... #path to fairseq
sacrebleu -t wmt18 -l en-de --echo src | $MOSES/tokenizer/normalize-punctuation.perl -l en \
  | $MOSES/tokenizer/tokenizer.perl -a -l en -q | python $BPEROOT/apply_bpe.py -c en2de_mt.bpe.code \
  |python $FAIRSEQ/interactive.py . --path en2de_mt.pt --remove-bpe --buffer-size 1024 --batch-size 8 \
   -s en -t de |grep -P '^H' |cut -f 3- | $MOSES/tokenizer/detokenizer.perl -l de -q \
  | sacrebleu -t wmt18 -l en-de

```



# Pre-training for abstractive summarization

To train the language model we used concatenated newscrawl and CNN-DailyMail dataset.


```
python train.py ${LM_DATA} -a bi_transformer_lm_big --clip-norm 0.1 --lr 0.0001 --dropout 0.1 \
  --max-tokens 2480 --no-progress-bar --log-interval 1 --criterion cross_entropy --fp16 \
  --optimizer nag --lr-scheduler cosine --warmup-init-lr 1e-07 --warmup-updates 16000 --min-lr 1e-09 \
  --distributed-port 12597 --distributed-world-size 32 --max-update 984000 --task language_modeling \
  --max-lr 1.0 --lr-period-updates 968000 --lr-shrink 1.0 --decoder-layers 12 --attention-dropout 0.1 \
  --decoder-embed-dim 1024 --ddp-backend no_c10d --sample-break-mode eos --skip-invalid-size-inputs-valid-test \
  --relu-dropout 0.05 --save-interval-updates 200000 --keep-interval-updates 10 \
  --save-dir ${LM_CHECKPOINT_PATH}

```

To train a final seq2seq model:

```
python train.py /private/home/edunov/cnn-dailymail/cnn-dailymail/finished_files/processed_nc_cnn --fp16 \
   --no-enc-token-positional-embeddings --elmo-affine --share-decoder-input-output-embed \
   --distributed-world-size 32 --distributed-port 17453 --no-progress-bar --max-update 30000 \
   --optimizer adam --adam-betas '(0.9, 0.98)' --skip-invalid-size-inputs-valid-test \
   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 \
   --ddp-backend no_c10d --min-lr 1e-09 --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 \
   --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --update-freq 4 --attention-dropout 0.2 \
   --elmo-dropout 0.2 --max-tokens 3584 --arch transformer_wmt_en_de --seed 1 --warmup-init-lr 1e-7 \
   --encoder-embed-path elmo:${LM_CHECKPOINT_PATH} --source-lang source --target-lang target
```


## To generate using the pre-trained model

```
curl https://dl.fbaipublicfiles.com/fairseq/models/pretraining/cnn_dailymail.tar.gz | tar xvzf -
cd cnn_dailymail
FAIRSEQ=... #path to fairseq

python $FAIRSEQ/generate.py . --path cnn_dailymail.pt --remove-bpe --gen-subset test \
   --batch-size 1 --min-len 60 --no-repeat-ngram 3 | tee cnn_dailymail.out

cat cnn_dailymail.out |grep -P '^H' | sort -nr -k1.2 | cut -f3- > cnn_dailymail.h.out

# This requires files2rouge to be installed: https://github.com/pltrdy/files2rouge
files2rouge cnn_dailymail.h.out ~/cnn-dailymail/model/cnn.txt

```



