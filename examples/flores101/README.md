<p align="center">
<img src="flores_logo.png" width="500">
</p>

# Flores101: Large-Scale Multilingual Machine Translation

## Introduction

Baseline pretrained models for small and large tracks of WMT 21 Large-Scale Multilingual Machine Translation competition.

Flores Task at WMT 21: http://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html

Flores announement blog post: https://ai.facebook.com/blog/flores-researchers-kick-off-multilingual-translation-challenge-at-wmt-and-call-for-compute-grants/



## Pretrained models

Model | Num layers | Embed dimension | FFN dimension| Vocab Size | #params | Download
---|---|---|---|---|---|---
`flores101_mm100_615M` | 12 | 1024 | 4096 | 256,000 | 615M | https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz
`flores101_mm100_175M` | 6 | 512 | 2048 | 256,000 | 175M | https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.tgz


These models are trained similar to [M2M-100](https://arxiv.org/abs/2010.11125) with additional support for the languages that are part of the WMT Large-Scale Multilingual Machine Translation track. Full list of languages can be found at the bottom.


## Example Generation code

### Download model, sentencepiece vocab

```bash
fairseq=/path/to/fairseq
cd $fairseq

# Download 615M param model.
wget https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz

# Extract 
tar -xvzf flores101_mm100_615M.tar.gz
```

### Encode using our SentencePiece Model
Note: Install SentencePiece from [here](https://github.com/google/sentencepiece)


```bash
fairseq=/path/to/fairseq
cd $fairseq

# Download example dataset From German to French
sacrebleu --echo src -l de-fr -t wmt19 | head -n 20 > raw_input.de-fr.de
sacrebleu --echo ref -l de-fr -t wmt19 | head -n 20 > raw_input.de-fr.fr

for lang in de fr ; do
    python scripts/spm_encode.py \
        --model flores101_mm100_615M/sentencepiece.bpe.model \
        --output_format=piece \
        --inputs=raw_input.de-fr.${lang} \
        --outputs=spm.de-fr.${lang}
done
```

### Binarization

```bash
fairseq-preprocess \
    --source-lang de --target-lang fr \
    --testpref spm.de-fr \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir data_bin \
    --srcdict flores101_mm100_615M/dict.txt --tgtdict flores101_mm100_615M/dict.txt
```

### Generation 


```bash
fairseq-generate \
    data_bin \
    --batch-size 1 \
    --path flores101_mm100_615M/model.pt \
    --fixed-dictionary flores101_mm100_615M/dict.txt \
    -s de -t fr \
    --remove-bpe 'sentencepiece' \
    --beam 5 \
    --task translation_multi_simple_epoch \
    --lang-pairs flores101_mm100_615M/language_pairs.txt \
    --decoder-langtok --encoder-langtok src \
    --gen-subset test \
    --fp16 \
    --dataset-impl mmap \
    --distributed-world-size 1 --distributed-no-spawn
```

### Supported Languages and lang code

Language | lang code
---|---
Akrikaans | af
Amharic | am
Arabic | ar
Assamese | as
Asturian | ast
Aymara | ay
Azerbaijani | az
Bashkir | ba
Belarusian | be
Bulgarian | bg
Bengali | bn
Breton | br
Bosnian | bs
Catalan | ca
Cebuano | ceb
Chokwe | cjk
Czech | cs
Welsh | cy
Danish | da
German | de
Dyula| dyu
Greek | el
English | en
Spanish | es
Estonian | et
Persian | fa
Fulah | ff
Finnish | fi
French | fr
Western Frisian | fy
Irish | ga
Scottish Gaelic | gd
Galician | gl
Gujarati | gu
Hausa | ha
Hebrew | he
Hindi | hi
Croatian | hr
Haitian Creole | ht
Hungarian | hu
Armenian | hy
Indonesian | id
Igbo | ig
Iloko | ilo
Icelandic | is
Italian | it
Japanese | ja
Javanese | jv
Georgian | ka
Kachin | kac
Kamba | kam
Kabuverdianu | kea
Kongo | kg
Kazakh | kk
Central Khmer | km
Kimbundu | kmb
Northern Kurdish | kmr
Kannada | kn
Korean | ko
Kurdish | ku
Kyrgyz | ky
Luxembourgish | lb
Ganda | lg
Lingala | ln
Lao | lo
Lithuanian | lt
Luo | luo
Latvian | lv
Malagasy | mg
Maori | mi
Macedonian | mk
Malayalam | ml
Mongolian | mn
Marathi | mr
Malay | ms
Maltese | mt
Burmese | my
Nepali | ne
Dutch | nl
Norwegian | no
Northern Sotho | ns
Nyanja | ny
Occitan | oc
Oromo | om
Oriya | or
Punjabi | pa
Polish | pl
Pashto | ps
Portuguese | pt
Quechua | qu
Romanian | ro
Russian | ru
Sindhi | sd
Shan | shn
Sinhala | si
Slovak | sk
Slovenian | sl
Shona | sn
Somali | so
Albanian | sq
Serbian | sr
Swati | ss
Sundanese | su
Swedish | sv
Swahili | sw
Tamil | ta
Telugu | te
Tajik | tg
Thai | th
Tigrinya | ti
Tagalog | tl
Tswana | tn
Turkish | tr
Ukrainian | uk
Umbundu | umb
Urdu | ur
Uzbek | uz
Vietnamese | vi
Wolof | wo
Xhosa | xh
Yiddish | yi
Yoruba | yo
Chinese| zh
Zulu | zu