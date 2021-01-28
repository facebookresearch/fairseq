# Cross-Lingual Language Model Pre-training

Below are some details for training Cross-Lingual Language Models (XLM) - similar to the ones presented in [Lample & Conneau, 2019](https://arxiv.org/pdf/1901.07291.pdf) - in Fairseq. The current implementation only supports the Masked Language Model (MLM) from the paper above.

## Downloading and Tokenizing Monolingual Data

Pointers to the monolingual data from wikipedia, used for training the XLM-style MLM model as well as details on processing (tokenization and BPE) it can be found in the [XLM Github Repository](https://github.com/facebookresearch/XLM#download--preprocess-monolingual-data).

Let's assume the following for the code snippets in later sections to work
- Processed data is in the folder: monolingual_data/processed
- Each language has 3 files for train, test and validation. For example we have the following files for English:
    train.en, valid.en
- We are training a model for 5 languages: Arabic (ar), German (de), English (en), Hindi (hi) and French (fr)
- The vocabulary file is monolingual_data/processed/vocab_mlm


## Fairseq Pre-processing and Binarization

Pre-process and binarize the data with the MaskedLMDictionary and cross_lingual_lm task

```bash
# Ensure the output directory exists
DATA_DIR=monolingual_data/fairseq_processed
mkdir -p "$DATA_DIR"

for lg in ar de en hi fr
do

  fairseq-preprocess \
  --task cross_lingual_lm \
  --srcdict monolingual_data/processed/vocab_mlm \
  --only-source \
  --trainpref monolingual_data/processed/train \
  --validpref monolingual_data/processed/valid \
  --testpref monolingual_data/processed/test \
  --destdir monolingual_data/fairseq_processed \
  --workers 20 \
  --source-lang $lg

  # Since we only have a source language, the output file has a None for the
  # target language. Remove this

  for stage in train test valid

    sudo mv "$DATA_DIR/$stage.$lg-None.$lg.bin" "$stage.$lg.bin"
    sudo mv "$DATA_DIR/$stage.$lg-None.$lg.idx" "$stage.$lg.idx"

  done

done
```

## Train a Cross-lingual Language Model similar to the XLM MLM model

Use the following command to train the model on 5 languages.

```
fairseq-train \
--task cross_lingual_lm monolingual_data/fairseq_processed \
--save-dir checkpoints/mlm \
--max-update 2400000 --save-interval 1 --no-epoch-checkpoints \
--arch xlm_base \
--optimizer adam --lr-scheduler reduce_lr_on_plateau \
--lr-shrink 0.5 --lr 0.0001 --stop-min-lr 1e-09 \
--dropout 0.1 \
--criterion legacy_masked_lm_loss \
--max-tokens 2048 --tokens-per-sample 256 --attention-dropout 0.1 \
--dataset-impl lazy --seed 0 \
--masked-lm-only \
--monolingual-langs 'ar,de,en,hi,fr' --num-segment 5 \
--ddp-backend=legacy_ddp
```

Some Notes:
- Using tokens_per_sample greater than 256 can cause OOM (out-of-memory) issues. Usually since MLM packs in streams of text, this parameter doesn't need much tuning.
- The Evaluation workflow for computing MLM Perplexity on test data is in progress.
- Finetuning this model on a downstream task is something which is not currently available.
