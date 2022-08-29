# No Language Left Behind : Evaluation

## Introduction
We are releasing all the translations of NLLB-200 (MoE, 54.5B) model on the following benchmarks:
- Translations of all 40602 directions in the [FLORES-200 benchmark](https://github.com/facebookresearch/flores/tree/main/flores200)
- Translations of non-FLORES benchmarks we evaluated  our model on in the [NLLB-200 paper](https://arxiv.org/abs/2207.04672). These include: Flores(v1), WAT, WMT, TICO, Mafand-mt and Autshumato. See details of each evaluation benchmark below.


## Downloading the translations:
- [FLORES-200 benchmark](https://github.com/facebookresearch/flores/tree/main/flores200) translations can be downloaded [here](https://tinyurl.com/nllbflorestranslations)
- The translations of the other benchmarks can be downloaded [here](https://tinyurl.com/nllbnonflorestranslations )

## Preparing the references:
- For FLORES-200 see [here](https://github.com/facebookresearch/flores/tree/main/flores200#download)
- For the other benchmarks:
    To prepare the references in a `$references_directory`, you will first need to manually downlaod some of the corpora and agree to terms of use.

    These include: 
    - Autshumato_MT_Evaluation_Set.zip from [here](https://repo.sadilar.org/handle/20.500.12185/506)
    - MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip from [here](https://sites.google.com/nyu.edu/madar/home#h.xpcfdhjyc95c)
    - 2017-01-mted-test.tgz from [here](https://wit3.fbk.eu/2017-01-b)
    - 2017-01-ted-test.tgz from [here](https://wit3.fbk.eu/2017-01-d)
    - 2015-01-test.tgz from [here](https://wit3.fbk.eu/2015-01-b)
    - 2014-01-test.tgz from [here](https://wit3.fbk.eu/2014-01-b)

    Note that due to copyright issues, we cannot release translations of IWSLT and MADAR but the download_other_corpora.py would allow you to reproduce our exact evaluation benchmark.

    The listed files should be in a `$pre_downloaded_resources` directory.

    You will also need some of the [MOSES scripts](https://github.com/moses-smt/mosesdecoder)
    ```    
    git clone https://github.com/moses-smt/mosesdecoder.git
    MOSES=PATH_to_mosesdecoder python examples/nllb/evaluation/download_other_corpora.py -d $references_directory -p $pre_downloaded_resources
    ```


## Scoring the translations
```
CORPUS=
METRIC=
python examples/nllb/evaluation/calculate_metrics.py \
    --corpus $CORPUS \
    --translate-dir ${generations} \
    --reference-dir ${references} \
    --metric $METRIC --output-dir ${output}
```

## Evaluation benchmarks:
Besides [FLORES-200](https://github.com/facebookresearch/flores/flores200), these are the evaluation benchmarks we evaluate NLLB-200 on:

- **Flores(v1)}**: with a total of 8 directions, the original Flores dataset~\citep{guzman2019floresv1} pairs four low-resource languages with English in the Wikimedia domain:
    - Nepali (ne, npi\_Deva)
    - Sinhala (si, sin\_Sinh)
    - Khmer (km, khm\_Khmr)
    - Pashto (ps, pbt\_Arab)

- **WAT**: we select 3 languages paired with English (6 directions) from the WAT competition:
    - (hin\_Deva)
    - (khm\_Khmr) 
    - (mya\_Mymr) 

- **WMT** : We evaluate on the 15 WMT languages selected in [Siddhant et al., 2020](https://aclanthology.org/2020.acl-main.252/). The 15 languages paired with English in this set are: 
    - Czech (WMT 18, cs, ces\_Latn)
    - German (WMT 14, de, deu\_Latn)
    - Estonian (WMT 18, et, est\_Latn)
    - Finnish (WMT 19, fi, fin\_Latn)
    - French (WMT 14, fr, fra\_Latn)
    - Gujarati (WMT 19, gu, guj\_Gujr)
    - Hindi (WMT 14, hi, hin\_Deva)
    - Kazakh (WMT 19, kk, kaz\_Cyrl)
    - Lithuanian (WMT 19, lt, lit\_Latn)
    - Standard Latvian (WMT 17, lv, lvs\_Latn)
    - Romanian (WMT 16, ro, ron\_Latn)
    - Russian (WMT 19, ru, rus\_Cyrl)
    - Spanish (WMT 13, es, spa\_Latn)
    - Turkish (WMT 18, tr, tur\_Latn)
    - Chinese (simplified) (WMT 19, zh, zho\_Hans).

- [**TICO**](https://tico-19.github.io/): sampled from a variety of public sources containing COVID-19 related content, this dataset comes from different domains (medical, news, conversational, etc.) and covers 36 languages. We pair 28 languages with English for a total of 56 directions.

- [**Mafand-MT**](https://github.com/masakhane-io/lafand-mt): an African news corpus that covers 16 languages. We evaluate 7 languages paired with English and 5 other paired with French for a total of 24 directions.
    - Paired with English (en, eng\_Latn)
        - Hausa (hau, hau\_Latn)
        - Igbo (ibo, ibo\_Latn)
        - Luganda (lug, lug\_Latn)
        - Swahili (swa, swh\_Latn)
        - Setswana (tsn, tsn\_Latn)
        - Yoruba (yor, yor\_Latn)
        - Zulu (zul, zul\_Latn)

    - Paired with French (fr, fra\_Latn)
        - Bambara (bam, bam\_Latn)
        - Ewe (ewe, ewe\_Latn)
        - Fon (fon, fon\_Latn)
        - Mossi (mos, mos\_Latn) 
        - Wolof (wol, wol\_Latn)

- [**Autshumato**](https://repo.sadilar.org/handle/20.500.12185/506): an evaluation set for machine translation of South African languages, it consists of 500 sentences from South African governmental data, translated separately by four different professional human translators for each of the 11 official South African languages. 9 of these languages are covered by NLLB-200:
    - Afrikaans (afr\_Latn)
    - English (eng\_Latn)
    - Sepedi / Northern Sotho (nso\_Latn)
    - Sesotho / Southern Sotho (sot\_Latn)
    - Siswati/Swati (ssw\_Latn)
    - Setswana/Tswana (tsn\_Latn)
    - Xitsonga/Tsonga (tso\_Latn)
    - IsiXhosa/Xhosa (xho\_Latn)
    - IsiZulu/Zulu (zul\_Latn). 

There is no standard valid/test split, so we use the first half (250 sentences yielding 1000 pairs) for validation and the second half for testing - see script.

