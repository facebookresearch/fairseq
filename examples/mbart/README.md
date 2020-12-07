# MBART: Multilingual Denoising Pre-training for Neural Machine Translation
[https://arxiv.org/abs/2001.08210]

## Introduction

MBART is a sequence-to-sequence denoising auto-encoder pre-trained on large-scale monolingual corpora in many languages using the BART objective. mBART is one of the first methods for pre-training a complete sequence-to-sequence model by denoising full texts in multiple languages, while previous approaches have focused only on the encoder, decoder, or reconstructing parts of the text.

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`mbart.CC25` | mBART model with 12 encoder and decoder layers trained on 25 languages' monolingual corpus | 610M | [mbart.CC25.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz)
`mbart.ft.ro_en` | finetune mBART cc25 model on ro-en language pairs | 610M | [mbart.cc25.ft.enro.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.ft.enro.tar.gz)

## Results

**[WMT16 EN-RO](https://www.statmt.org/wmt16/translation-task.html)**

_(test set, no additional data used)_

Model | en-ro | ro-en
---|---|---
`Random` | 34.3 | 34.0
`mbart.cc25` | 37.7 | 37.8
`mbart.enro.bilingual` | 38.5 | 38.5 

## BPE data
# download model
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz
tar -xzvf mbart.CC25.tar.gz
# bpe data
install SPM [here](https://github.com/google/sentencepiece)
```bash
SPM=/path/to/sentencepiece/build/src/spm_encode
MODEL=sentence.bpe.model
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} &
```

## Preprocess data

```bash
DICT=dict.txt
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/${TRAIN}.spm \
  --validpref ${DATA}/${VALID}.spm \
  --testpref ${DATA}/${TEST}.spm \
  --destdir ${DEST}/${NAME} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70
```

## Finetune on EN-RO
Finetune on mbart CC25

```bash
PRETRAIN=mbart.cc25 # fix if you moved the downloaded checkpoint
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

fairseq-train path_2_data \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang en_XX --target-lang ro_RO \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2 \
  --restore-file $PRETRAIN \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs $langs \
  --ddp-backend no_c10d
```
## Generate on EN-RO
Get sacrebleu on finetuned en-ro model

get tokenizer  [here](https://github.com/rsennrich/wmt16-scripts)
```bash  
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.ft.enro.tar.gz  
tar -xzvf mbart.cc25.ft.enro.tar.gz
```

```bash
model_dir=MBART_finetuned_enro # fix if you moved the checkpoint

fairseq-generate path_2_data \
  --path $model_dir/model.pt \
  --task translation_from_pretrained_bart \
  --gen-subset test \
  -t ro_RO -s en_XX \
  --bpe 'sentencepiece' --sentencepiece-model $model_dir/sentence.bpe.model \
  --sacrebleu --remove-bpe 'sentencepiece' \
  --batch-size 32 --langs $langs > en_ro

cat en_ro | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > en_ro.hyp
cat en_ro | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g' |$TOKENIZER ro > en_ro.ref
sacrebleu -tok 'none' -s 'none' en_ro.ref < en_ro.hyp
```

## Citation

```bibtex
@article{liu2020multilingual,
    title={Multilingual Denoising Pre-training for Neural Machine Translation},
    author={Yinhan Liu and Jiatao Gu and Naman Goyal and Xian Li and Sergey Edunov and Marjan Ghazvininejad and Mike Lewis and Luke Zettlemoyer},
    year={2020},
    eprint={2001.08210},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
