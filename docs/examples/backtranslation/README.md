# Understanding Back-Translation at Scale (Edunov et al., 2018)

This page includes pre-trained models from the paper [Understanding Back-Translation at Scale (Edunov et al., 2018)](https://arxiv.org/abs/1808.09381).

## Pre-trained models

Model | Description | Dataset | Download
---|---|---|---
`transformer.wmt18.en-de` | Transformer <br> ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381)) <br> WMT'18 winner | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz) <br> See NOTE in the archive

## Example usage (torch.hub)

We require a few additional Python dependencies for preprocessing:
```bash
pip install subword_nmt sacremoses
```

Then to generate translations from the full model ensemble:
```python
import torch

# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt18.en-de', ... ]

# Load the WMT'18 En-De ensemble
en2de_ensemble = torch.hub.load(
    'pytorch/fairseq', 'transformer.wmt18.en-de',
    checkpoint_file='wmt18.model1.pt:wmt18.model2.pt:wmt18.model3.pt:wmt18.model4.pt:wmt18.model5.pt',
    tokenizer='moses', bpe='subword_nmt')

# The ensemble contains 5 models
len(en2de_ensemble.models)
# 5

# Translate
en2de_ensemble.translate('Hello world!')
# 'Hallo Welt!'
```

## Training your own model (WMT'18 English-German)

The following instructions can be adapted to reproduce the models from the paper.


#### Step 1. Prepare parallel data and optionally train a baseline (English-German) model

First download and preprocess the data:
```bash
# Download and prepare the data
cd examples/backtranslation/
bash prepare-wmt18en2de.sh
cd ../..

# Binarize the data
TEXT=examples/backtranslation/wmt18_en_de
fairseq-preprocess \
    --joined-dictionary \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt18_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

# Copy the BPE code into the data-bin directory for future use
cp examples/backtranslation/wmt18_en_de/code data-bin/wmt18_en_de/code
```

(Optionally) Train a baseline model (English-German) using just the parallel data:
```bash
CHECKPOINT_DIR=checkpoints_en_de_parallel
fairseq-train --fp16 \
    data-bin/wmt18_en_de \
    --source-lang en --target-lang de \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 3584 --update-freq 16 \
    --max-update 30000 \
    --save-dir $CHECKPOINT_DIR
# Note: the above command assumes 8 GPUs. Adjust `--update-freq` if you have a
# different number of GPUs.
```

Average the last 10 checkpoints:
```bash
python scripts/average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --num-epoch-checkpoints 10 \
    --output $CHECKPOINT_DIR/checkpoint.avg10.pt
```

Evaluate BLEU:
```bash
# tokenized BLEU on newstest2017:
bash examples/backtranslation/tokenized_bleu.sh \
    wmt17 \
    en-de \
    data-bin/wmt18_en_de \
    data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint.avg10.pt
# BLEU4 = 29.57, 60.9/35.4/22.9/15.5 (BP=1.000, ratio=1.014, syslen=63049, reflen=62152)
# compare to 29.46 in Table 1, which is also for tokenized BLEU

# generally it's better to report (detokenized) sacrebleu though:
bash examples/backtranslation/sacrebleu.sh \
    wmt17 \
    en-de \
    data-bin/wmt18_en_de \
    data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint.avg10.pt
# BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt17+tok.13a+version.1.4.3 = 29.0 60.6/34.7/22.4/14.9 (BP = 1.000 ratio = 1.013 hyp_len = 62099 ref_len = 61287)
```


#### Step 2. Back-translate monolingual German data

Train a reverse model (German-English) to do the back-translation:
```bash
CHECKPOINT_DIR=checkpoints_de_en_parallel
fairseq-train --fp16 \
    data-bin/wmt18_en_de \
    --source-lang de --target-lang en \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 3584 --update-freq 16 \
    --max-update 30000 \
    --save-dir $CHECKPOINT_DIR
# Note: the above command assumes 8 GPUs. Adjust `--update-freq` if you have a
# different number of GPUs.
```

Let's evaluate the back-translation (BT) model to make sure it is well trained:
```bash
bash examples/backtranslation/sacrebleu.sh \
    wmt17 \
    de-en \
    data-bin/wmt18_en_de \
    data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint_best.py
# BLEU+case.mixed+lang.de-en+numrefs.1+smooth.exp+test.wmt17+tok.13a+version.1.4.3 = 34.9 66.9/41.8/28.5/19.9 (BP = 0.983 ratio = 0.984 hyp_len = 63342 ref_len = 64399)
# compare to the best system from WMT'17 which scored 35.1: http://matrix.statmt.org/matrix/systems_list/1868
```

Next prepare the monolingual data:
```bash
# Download and prepare the monolingual data
# By default the script samples 25M monolingual sentences, which after
# deduplication should be just over 24M sentences. These are split into 25
# shards, each with 1M sentences (except for the last shard).
cd examples/backtranslation/
bash prepare-de-monolingual.sh
cd ../..

# Binarize each shard of the monolingual data
TEXT=examples/backtranslation/wmt18_de_mono
for SHARD in $(seq -f "%02g" 0 24); do \
    fairseq-preprocess \
        --only-source \
        --source-lang de --target-lang en \
        --joined-dictionary \
        --srcdict data-bin/wmt18_en_de/dict.de.txt \
        --testpref $TEXT/bpe.monolingual.dedup.${SHARD} \
        --destdir data-bin/wmt18_de_mono/shard${SHARD} \
        --workers 20; \
    cp data-bin/wmt18_en_de/dict.en.txt data-bin/wmt18_de_mono/shard${SHARD}/; \
done
```

Now we're ready to perform back-translation over the monolingual data. The
following command generates via sampling, but it's possible to use greedy
decoding (`--beam 1`), beam search (`--beam 5`),
top-k sampling (`--sampling --beam 1 --sampling-topk 10`), etc.:
```bash
mkdir backtranslation_output
for SHARD in $(seq -f "%02g" 0 24); do \
    fairseq-generate --fp16 \
        data-bin/wmt18_de_mono/shard${SHARD} \
        --path $CHECKPOINT_DIR/checkpoint_best.pt \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 4096 \
        --sampling --beam 1 \
    > backtranslation_output/sampling.shard${SHARD}.out; \
done
```

After BT, use the `extract_bt_data.py` script to re-combine the shards, extract
the back-translations and apply length ratio filters:
```bash
python examples/backtranslation/extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output backtranslation_output/bt_data --srclang en --tgtlang de \
    backtranslation_output/sampling.shard*.out

# Ensure lengths are the same:
# wc -l backtranslation_output/bt_data.{en,de}
#   21795614 backtranslation_output/bt_data.en
#   21795614 backtranslation_output/bt_data.de
#   43591228 total
```

Binarize the filtered BT data and combine it with the parallel data:
```bash
TEXT=backtranslation_output
fairseq-preprocess \
    --source-lang en --target-lang de \
    --joined-dictionary \
    --srcdict data-bin/wmt18_en_de/dict.en.txt \
    --trainpref $TEXT/bt_data \
    --destdir data-bin/wmt18_en_de_bt \
    --workers 20

# We want to train on the combined data, so we'll symlink the parallel + BT data
# in the wmt18_en_de_para_plus_bt directory. We link the parallel data as "train"
# and the BT data as "train1", so that fairseq will combine them automatically
# and so that we can use the `--upsample-primary` option to upsample the
# parallel data (if desired).
PARA_DATA=$(readlink -f data-bin/wmt18_en_de)
BT_DATA=$(readlink -f data-bin/wmt18_en_de_bt)
COMB_DATA=data-bin/wmt18_en_de_para_plus_bt
mkdir -p $COMB_DATA
for LANG in en de; do \
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA}/train.en-de.$LANG.$EXT; \
        ln -s ${BT_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA}/train1.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid.en-de.$LANG.$EXT ${COMB_DATA}/valid.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/test.en-de.$LANG.$EXT ${COMB_DATA}/test.en-de.$LANG.$EXT; \
    done; \
done
```


#### 3. Train an English-German model over the combined parallel + BT data

Finally we can train a model over the parallel + BT data:
```bash
CHECKPOINT_DIR=checkpoints_en_de_parallel_plus_bt
fairseq-train --fp16 \
    data-bin/wmt18_en_de_para_plus_bt \
    --upsample-primary 16 \
    --source-lang en --target-lang de \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 3584 --update-freq 16 \
    --max-update 100000 \
    --save-dir $CHECKPOINT_DIR
# Note: the above command assumes 8 GPUs. Adjust `--update-freq` if you have a
# different number of GPUs.
```

Average the last 10 checkpoints:
```bash
python scripts/average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --num-epoch-checkpoints 10 \
    --output $CHECKPOINT_DIR/checkpoint.avg10.pt
```

Evaluate BLEU:
```bash
# tokenized BLEU on newstest2017:
bash examples/backtranslation/tokenized_bleu.sh \
    wmt17 \
    en-de \
    data-bin/wmt18_en_de \
    data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint.avg10.pt
# BLEU4 = 32.35, 64.4/38.9/26.2/18.3 (BP=0.977, ratio=0.977, syslen=60729, reflen=62152)
# compare to 32.35 in Table 1, which is also for tokenized BLEU

# generally it's better to report (detokenized) sacrebleu:
bash examples/backtranslation/sacrebleu.sh \
    wmt17 \
    en-de \
    data-bin/wmt18_en_de \
    data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint.avg10.pt
# BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt17+tok.13a+version.1.4.3 = 31.5 64.3/38.2/25.6/17.6 (BP = 0.971 ratio = 0.971 hyp_len = 59515 ref_len = 61287)
```


## Citation
```bibtex
@inproceedings{edunov2018backtranslation,
  title = {Understanding Back-Translation at Scale},
  author = {Edunov, Sergey and Ott, Myle and Auli, Michael and Grangier, David},
  booktitle = {Conference of the Association for Computational Linguistics (ACL)},
  year = 2018,
}
```
