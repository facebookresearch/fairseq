# Multilingual Translation

[[Multilingual Translation with Extensible Multilingual Pretraining and Finetuning, https://arxiv.org/abs/2008.00401]](https://arxiv.org/abs/2008.00401)

## Introduction

This work is for training multilingual translation models with multiple bitext datasets. This multilingual translation framework supports (see [[training section]](#Training) and [[finetuning section]](#Finetuning) for examples)

* temperature based sampling over unbalancing datasets of different translation directions
  - --sampling-method' with
            choices=['uniform', 'temperature',  'concat']
  - --sampling-temperature
* configurable to automatically add source and/or target language tokens to source/target sentences using data which are prepared in the same way as bilignual training
  - --encoder-langtok with choices=['src', 'tgt', None] to specify whether to add source or target language tokens to the source sentences
  - --decoder-langtok (binary option) to specify whether to add target language tokens to the target sentences or not
* finetuning mBART pretrained models for multilingual translation
  - --finetune-from-model to specify the path from which to load the pretrained model

## Preprocessing data
Multilingual training requires a joint BPE vocab. Please follow [mBART's preprocessing steps](https://github.com/pytorch/fairseq/tree/master/examples/mbart#bpe-data) to reuse our pretrained sentence-piece model.

You can also train a joint BPE model on your own dataset and then follow the steps in [[link]](https://github.com/pytorch/fairseq/tree/master/examples/translation#multilingual-translation).

## Training


```bash
lang_pairs=<language pairs to be trained, e.g. "en-cs,cs-en">
path_2_data=<set to data path>
lang_list=<a file which contains a list of languages separated by new lines>

fairseq-train $path_2_data \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1.5 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-05 --min-lr -1 --warmup-updates 2500 --max-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2
```

## Finetuning
We can also finetune multilingual models from a monolingual pretrained models, e.g. [mMBART](https://github.com/pytorch/fairseq/tree/master/examples/mbart).
```bash
lang_pairs=<language pairs to be trained, e.g. "en-cs,cs-en">
path_2_data=<set to data path>
lang_list=<a file which contains a list of languages separated by new lines>
pretrained_model=<path to the pretrained model, e.g. mbart or another trained multilingual model>

fairseq-train $path_2_data \
  --finetune-from-model $pretrained_model \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1.5 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-05 --min-lr -1 --warmup-updates 2500 --max-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2
```
## Generate
The following command uses the multilingual task (translation_multi_simple_epoch) to generate translation  from $source_lang to $target_lang on the test dataset. During generaton, the source language tokens are added to source sentences and the target language tokens are added as the starting token to decode target sentences. Options --lang-dict and --lang-pairs are needed to tell the generation process the ordered list of languages and translation directions that the trained model are awared of; they will need to be consistent with the training.

```bash
model=<multilingual model>
source_lang=<source language>
target_lang=<target language>

fairseq-generate $path_2_data \
  --path $model \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang $source_lang \
  --target-lang $target_lang
  --sacrebleu --remove-bpe 'sentencepiece'\
  --max-sentences 32 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" > ${source_lang}_${target_lang}.txt
```
Fairseq will generate translation into a file {source_lang}_${target_lang}.txt with sacreblue at the end.

You can also use costomized tokenizer to compare the performance with the literature. For example, you get a tokenizer [here](https://github.com/rsennrich/wmt16-scripts) and do the following:
```bash
TOKENIZER=<path to a customized tokenizer for decoding evaluation>
TOK_CMD=<"$TOKENIZER $target_lang" or cat for sacrebleu>

cat {source_lang}_${target_lang}.txt | grep -P "^H" |sort -V |cut -f 3- |$TOK_CMD > ${source_lang}_${target_lang}.hyp
cat {source_lang}_${target_lang}.txt | grep -P "^T" |sort -V |cut -f 2- |$TOK_CMD > ${source_lang}_${target_lang}.ref
sacrebleu -tok 'none' -s 'none' ${source_lang}_${target_lang}.ref < ${source_lang}_${target_lang}.hyp
```



## Citation

```bibtex
@article{tang2020multilingual,
    title={Multilingual Translation with Extensible Multilingual Pretraining and Finetuning},
    author={Yuqing Tang and Chau Tran and Xian Li and Peng-Jen Chen and Naman Goyal and Vishrav Chaudhary and Jiatao Gu and Angela Fan},
    year={2020},
    eprint={2008.00401},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
