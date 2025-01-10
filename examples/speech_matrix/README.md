# SpeechMatrix: A Large-Scale Mined Corpus of Multilingual Speech-to-Speech Translations


### Installation

1. [fairseq](../../README.md)
2. [speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis)


## Speech-to-Speech Mined Data

SpeechMatrix provides massive parallel speech which is mined from [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) in 17 languages: Czech (cs), German (de), English (en), Spanish (es), Estonian (et), Finnish (fi), French (fr), Croatian (hr), Hungarian (hu), Italian (it), Lithuanian (lt), Dutch (nl), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk) and Slovenian (sl).

Here is a summary of mined speech-to-speech data statistics, and the duration (hours) of source speech is reported in each of 272 language directions.

Src/Tgt   |  cs   |  de   |  en   |  es   |  et   |  fi   |  fr   |  hr   |  hu   |  it   |  lt   |  nl   |  pl   |  pt   |  ro   |  sk   |  sl
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
cs   | - | 2381 | 3208 | 2290 | 952 | 1312 | 2476 | 726 | 1396 | 2410 | 84 | 2377 | 2516 | 1867 | 1190 | 2146 | 452
 de   | 2386 | - | 4734 | 3113 | 901 | 1477 | 3536 | 498 | 1871 | 3476 | 41 | 3384 | 2632 | 2250 | 1281 | 1646 | 361
 en   | 3172 | 4676 | - | 4715 | 1585 | 2169 | 5178 | 824 | 2266 | 4897 | 82 | 4422 | 3583 | 3572 | 2258 | 2306 | 586
 es   | 2240 | 3041 | 4708 | - | 862 | 1373 | 4446 | 528 | 1599 | 4418 | 47 | 3067 | 2646 | 3484 | 1857 | 1603 | 308
 et   | 943 | 892 | 1593 | 877 | - | 1201 | 934 | 265 | 1119 | 1019 | 39 | 1055 | 949 | 721 | 419 | 780 | 196
 fi   | 1296 | 1463 | 2180 | 1393 | 1197 | - | 1449 | 306 | 1473 | 1599 | 47 | 1654 | 1350 | 1128 | 621 | 977 | 260
 fr   | 2424 | 3457 | 5171 | 4455 | 923 | 1435 | - | 560 | 1711 | 4618 | 50 | 3273 | 2822 | 3384 | 1991 | 1657 | 326
 hr   | 736 | 507 | 854 | 553 | 273 | 317 | 588 | - | 328 | 615 | 24 | 546 | 660 | 433 | 277 | 586 | 136
 hu   | 1417 | 1897 | 2346 | 1672 | 1140 | 1507 | 1787 | 328 | - | 1855 | 68 | 1839 | 1566 | 1315 | 808 | 1064 | 311
 it   | 2404 | 3460 | 4948 | 4500 | 1028 | 1614 | 4700 | 607 | 1823 | - | 103 | 3414 | 2848 | 3421 | 1995 | 1656 | 474
 lt   | 78 | 38 | 79 | 46 | 37 | 44 | 48 | 21 | 61 | 95 | - | 77 | 80 | 35 | 18 | 64 | 6
 nl   | 2322 | 3305 | 4396 | 3066 | 1040 | 1633 | 3269 | 521 | 1768 | 3355 | 80 | - | 2459 | 2399 | 1352 | 1646 | 458
 pl   | 2530 | 2646 | 3662 | 2735 | 967 | 1378 | 2913 | 656 | 1554 | 2883 | 88 | 2540 | - | 2121 | 1301 | 1892 | 431
 pt   | 1849 | 2224 | 3606 | 3525 | 722 | 1131 | 3421 | 421 | 1279 | 3403 | 37 | 2436 | 2087 | - | 1579 | 1358 | 247
 ro   | 1187 | 1275 | 2290 | 1894 | 423 | 627 | 2024 | 271 | 789 | 1996 | 19 | 1384 | 1288 | 1592 | - | 870 | 125
 sk   | 2127 | 1628 | 2329 | 1631 | 781 | 982 | 1685 | 574 | 1038 | 1650 | 69 | 1676 | 1869 | 1361 | 867 | - | 370
 sl   | 436 | 350 | 579 | 307 | 192 | 254 | 324 | 128 | 295 | 461 | 6 | 454 | 413 | 241 | 121 | 359 | -


### Speech-to-Speech Alignments

The mined data has speech alignments at the threshold of 1.06. For the train set, its overlap with VoxPopuli valid and test data has been removed.

```bash
# SAVE_ROOT: the directory to save mined data
python mined_train_sets/download_mined_data.py \
    --save-root ${SAVE_ROOT}
```

Audios are saved to ${SAVE_ROOT}/audios/. For example, English audios are compressed into en_aud.zip.

Speech alignments are saved to ${SAVE_ROOT}/aligned_speech/. For example, en-fr.tsv.gz contains a pair of aligned audio paths in English and French respectively together with their alignment score in each line.

## Speech Transcriptions

While SpeechMatrix focuses on speech-only data mining and translation, we provide transcriptions for the mined speech in case they are needed for future research. The transcriptions are generated with [Whisper](https://github.com/openai/whisper), we use medium.en for English transcribing, and medium for other langauges. Curently transcriptions are provided the target speech in these language directions: {"cs", "de", "en", "es", "et", "fi", "fr", "hu", "it", "lt", "nl", "pl", "pt", "ro", "sl"}-{"de", "en", "es", "fr", "nl"}.

```bash
# SAVE_ROOT: the directory to save mined data
python mined_train_sets/download_transcriptions.py \
    --save-root ${SAVE_ROOT}
```

## Speech-to-Unit Data

Speech-to-unit manifests are provided to facilitate S2U training, which has units extracted from [HuBERT models](valid_test_sets/README.md). VoxPopuli test and valid manifests are also released.


```bash
# SAVE_ROOT: the directory to save mined data
python mined_train_sets/download_speech_to_unit.py \
    --save-root ${SAVE_ROOT}
```

The speech-to-unit data for a language pair is saved to ```${SAVE_ROOT}/s2u_manifests/${lang_pair}```

### Reproduce Bilingual Train Data

In our bilingual speech-to-unit experiments, we set different thresholds to select a subset of mined data for the purpose of training efficiency. You can reproduce the training data and configurations as below:

```bash
# SAVE_ROOT: the directory to save SpeechMatrix mined data
python3 speech_to_speech/prep_bilingual_textless_manifest.py --save-root ${SAVE_ROOT}
```

##  Speech-to-Speech Valid and Test Data

For speech-to-speech training, we use mined data as the train set, and prepare valid and test sets using [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli), [EuroParl-ST](https://www.mllp.upv.es/europarl-st/) and [FLEURS](https://huggingface.co/datasets/google/fleurs). Check out [here](valid_test_sets/README.md) for valid and test data preparations.


## Speech-to-Speech Training

### Textless Model

Check out [here](../speech_to_speech/docs/direct_s2st_discrete_units.md) for Textless S2U model training.


#### Bilingual Textless Model Release
Src/Tgt | de | en | es | fr | it | nl | pl | pt | ro
|---|---|---|---|---|---|---|---|---|---
cs | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_cs_ro.pt)
de | - | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_de_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_de_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_de_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_de_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_de_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_de_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_de_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_de_ro.pt)
en | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_en_de.pt) | - | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_en_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_en_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_en_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_en_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_en_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_en_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_en_ro.pt)
es | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_es_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_es_en.pt) | - | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_es_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_es_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_es_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_es_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_es_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_es_ro.pt)
et | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_et_ro.pt)
fi | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fi_ro.pt)
fr | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fr_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fr_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fr_es.pt) | - | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fr_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fr_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fr_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fr_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_fr_ro.pt)
hr | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hr_ro.pt)
hu | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_hu_ro.pt)
it | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_it_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_it_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_it_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_it_fr.pt) | - | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_it_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_it_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_it_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_it_ro.pt)
nl | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_nl_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_nl_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_nl_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_nl_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_nl_it.pt) | - | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_nl_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_nl_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_nl_ro.pt)
pl | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pl_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pl_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pl_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pl_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pl_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pl_nl.pt) | - | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pl_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pl_ro.pt)
pt | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pt_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pt_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pt_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pt_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pt_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pt_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pt_pl.pt) | - | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_pt_ro.pt)
ro | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_ro_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_ro_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_ro_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_ro_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_ro_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_ro_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_ro_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_ro_pt.pt) | -
sk | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sk_ro.pt)
sl | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_de.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_en.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_es.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_fr.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_it.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_nl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_pl.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_pt.pt) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_sl_ro.pt)

### Multilingual Textless Model

Download 260M Slavic-to-English [model](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_textless_slavic_en_260m.pt).

### XM Transformer
Check out [here](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/enhanced_direct_s2st_discrete_units.md) for XM Transformer model training.

Example training command:
```
python3 $FAIRSEQ_ROOT/train.py --distributed-world-size 32 --distributed-port 12314 --save-dir $SAVE_DIR $DATA_ROOT --ddp-backend legacy_ddp --num-workers 0 --task speech_to_text --criterion label_smoothed_cross_entropy --no-epoch-checkpoints --report-accuracy --clip-norm 10.0 --log-format simple --log-interval 500 --seed 121 --max-update 160000 --share-decoder-input-output-embed --validate-interval 1 --save-interval 1 --save-interval-updates 500 --skip-invalid-size-inputs-valid-test --keep-best-checkpoints 10 --optimizer adam --adam-betas '"'"'(0.9, 0.98)'"'"' --lr 0.0001 --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --lr-scheduler inverse_sqrt --warmup-updates 5000 --arch xm_transformer --normalize --adaptor-n-layers 1 --decoder-attention-heads 16 --decoder-normalize-before --load-pretrained-decoder-from ${UNIT_MBART_PATH} --w2v-path ${XLSR_PATH} --config-yaml config.yaml --mask-prob 0.3 --mask-channel-prob 0.25 --mask-channel-length 10 --layerdrop 0.1 --finetune-decoder-params all --label-smoothing 0.2 --patience 30 --max-tokens 9000 --max-tokens-valid 9000 --max-target-positions 9000 --max-source-positions 9000 --max-positions 9000 --update-freq 2 --train-subset $TRAIN_SUBSET --valid-subset $VALID_SUBSET --checkpoint-activations --encoder-proj
```

Architecture | direction | #params | Link
|---|---|---|---
Dense XM| Slavic-to-English | 1.3B | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_xm_transformer_slavic_en_1.3b.pt)
Dense XM| All-to-English | 1.3B | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_xm_transformer_all_en_1.3b.pt)


### Sparse XM Transformer (gshard)
Following arguments could be added for gshard in XM Transformer training, following the settings in paper (decoder-only MoE):

```
--moe-freq 2 --encoder-moe-freq 0 --decoder-moe-freq 2 --moe-expert-count $NUM_EXPERTS
```

Architecture | direction | #params | Link
|---|---|---|---
gshard XM| Slavic-to-English | 4.3B | [ckpts](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_gshard_slavic_en_4.3b.tar)
gshard XM| All-to-English | 4.3B | [ckpts](https://dl.fbaipublicfiles.com/speech_matrix/s2s_models/checkpoint_gshard_all_en_4.3b.tar)


## Inference

Download vocoders below to synthesize target speech from predicted unit sequence.

### Unit-based HiFi-GAN vocoder
Unit config | Unit size | Vocoder language | Dataset | Model
|---|---|---|---|---
mHuBERT, layer 11 | 1000 | de | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_de.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_de.json)
mHuBERT, layer 11 | 1000 | nl | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_nl.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_nl.json)
mHuBERT, layer 11 | 1000 | fi | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_fi.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_fi.json)
mHuBERT, layer 11 | 1000 | hu | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_hu.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_hu.json)
mHuBERT, layer 11 | 1000 | et | [Common Voice](https://github.com/common-voice/cv-dataset) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_et.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_et.json)
mHuBERT, layer 11 | 800 | it | [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_it.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_it.json)
mHuBERT, layer 11 | 1000 | pt | [Common Voice](https://github.com/common-voice/cv-dataset) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_pt.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_pt.json)
mHuBERT, layer 11 | 1000 | ro | [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_ro.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_ro.json)
mHuBERT, layer 11 | 1000 | cs | [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_cs.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_cs.json)
mHuBERT, layer 11 | 1000 | pl | [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_pl.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_pl.json)
mHuBERT, layer 11 | 1000 | hr | [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_hr.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_hr.json)
mHuBERT, layer 11 | 1000 | lt | [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_lt.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_lt.json)
mHuBERT, layer 11 | 1000 | sk | [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_sk.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_sk.json)
mHuBERT, layer 11 | 1000 | sl | [VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main/voxpopuli) | [ckpt](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/vocoder_sl.pt), [config](https://dl.fbaipublicfiles.com/speech_matrix/vocoder/config_sl.json)


For English (en), Spanish (es) and French (fr), we reuse HuBERT and kmeans models released by [textless model](../speech_to_speech/docs/textless_s2st_real_data.md).



### Textless Model Inference


Check out [here](../speech_to_speech/docs/direct_s2st_discrete_units.md) for Textless model inference.


### XM Transformer Model Inference
1. Check out the inference in [fairseq-S2T](../speech_to_speech/docs/enhanced_direct_s2st_discrete_units.md) to generate unit sequences (`${RESULTS_PATH}/generate-${GEN_SUBSET}.txt`).

```
fairseq-generate $DATA_ROOT \
  --config-yaml config.yaml \
  --task speech_to_text  \
  --path $MODEL_DIR/checkpoint_best.pt  --gen-subset $GEN_SUBSET \
  --max-tokens 18000 \
  --beam 10 --max-len-a 0.003125 --max-len-b 200 \
  --results-path ${RESULTS_PATH}
```

For MoE inference, add the following options and make sure

(1) #num_experts % #num_gpus == 0

(2) No OOM issue
```
--is-moe --distributed-world-size $NUM_GPUS
```

2. Convert unit sequences to waveform.

```
grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit

python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${RESULTS_PATH} --dur-prediction
```

## ASR-BLEU evaluation

Use the following command to compute ASR-BLEU with inference results on test sets

TGT_LANG: Target language

AUDIO_FOLDER: A folder contains all inference results (audio files)

REFERENCE_PATH: A txt file with each line to be translation result in plain text

```
cd ${FAIRSEQ_ROOT}/examples/speech_to_speech/asr_bleu
python3 compute_asr_bleu.py --lang ${TGT_LANG} \
--audio_dirpath ${AUDIO_FOLDER} \
--reference_path ${REFERENCE_PATH} \
--reference_format txt
```

## Mining with Speech Encoder

For people who are interested in trying out our speech encoders and mining parallel speech by themselves, we also release speech encoders. Please check out [speech encoding intructions](speech_laser_encoders.md) for more details.

## Huggingface Demo for SpeechMatrix models
This demo on huggingface has all-en multilingual model and bilingual models with target languages {en,fr,es} trained with SpeechMatrix data. https://huggingface.co/spaces/facebook/speech_matrix

## Citation

```
@inproceedings{speech-matrix,
    title = "{S}peech{M}atrix: A Large-Scale Mined Corpus of Multilingual Speech-to-Speech Translations",
    author = "Paul-Ambroise Duquenne and
      Hongyu Gong and
      Ning Dong and
      Jingfei Du and
      Ann Lee and
      Vedanuj Goswani and
      Changhan Wang and
      Juan Pino and
      Beno Sagot and
      Holger Schwenk",
}
```

## License

The released models and dataset are under [CC-BY-NC 4.0](../../LICENSE.model).
