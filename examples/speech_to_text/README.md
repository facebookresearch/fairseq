# Speech-to-Text (S2T) Modeling

[https://arxiv.org/abs/2010.05171](https://arxiv.org/abs/2010.05171)

Examples for speech recognition (ASR) and speech-to-text translation (ST) with fairseq.

## Data Preparation
S2T modeling data consists of source speech features, target text and other optional information
(source text, speaker id, etc.). Fairseq S2T uses per-dataset-split TSV manifest files
to store these information. Each data field is represented by a column in the TSV file.

Unlike text token embeddings, speech features (e.g. log mel-scale filter banks) are usually fixed
during model training and can be pre-computed. The manifest file contains the path to
either the feature file in NumPy format or the WAV/FLAC audio file. For the latter,
features will be extracted on-the-fly by fairseq S2T. Optionally, feature/audio files can be packed
into uncompressed ZIP files (then accessed via byte offset and length) to improve I/O performance.

Fairseq S2T also employs a YAML file for data related configurations: tokenizer type and dictionary path
for the target text, feature transforms such as CMVN (cepstral mean and variance normalization) and SpecAugment,
temperature-based resampling, etc.

## Model Training & Evaluation
Fairseq S2T uses the unified `fairseq-train`/`fairseq-generate` interface for model training and evaluation.
It requires arguments `--task speech_to_text` and `--arch <arch in fairseq.models.speech_to_text.*>`.


## Example 1: Speech Recognition (ASR) on LibriSpeech

#### Data preparation
Download and preprocess [LibriSpeech](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf) data with
```bash
python examples/speech_to_text/prep_librispeech_data.py --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
```
where `LS_ROOT` is the root path for downloaded data as well as generated manifest and feature files.

#### Training
```bash
fairseq-train ${LS_ROOT} --train-subset train --valid-subset dev --save-dir ${SAVE_DIR} --num-workers 4 \
    --max-tokens 40000 --task speech_to_text --criterion label_smoothed_cross_entropy --max-update 300000 \
    --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --clip-norm 10.0 --seed 1 --update-freq 8
```
where `SAVE_DIR` is the checkpoint root path. Here we use `--arch s2t_transformer_s` (31M parameters) as example.
You may switch to `s2t_transformer_m` (71M) or `s2t_transformer_l` (268M) for better performance. We set
`--update-freq 8` to simulate 8 GPUs with 1 GPU. You may want to update it accordingly when using more than 1 GPU.

#### Inference & Evaluation
Average the last 10 checkpoints and evaluate on the 4 splits
(`dev-clean`, `dev-other`, `test-clean` and `test-other`):
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} --num-epoch-checkpoints 10 \
    --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
for SUBSET in dev-clean dev-other test-clean test-other; do
    fairseq-generate ${LS_ROOT} --gen-subset ${SUBSET} --task speech_to_text \
        --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 --scoring wer
done
```

#### Result

| --arch | Params | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|---|
| s2t_transformer_s | 30M | 4.1 | 9.3 | 4.4 | 9.2 |
| s2t_transformer_sp | 35M | 3.9 | 9.3 | 4.3 | 8.8 |
| s2t_transformer_m | 71M | 3.5 | 8.1 | 3.7 | 8.1 |
| s2t_transformer_mp | 84M | 3.3 | 7.8 | 3.7 | 8.2 |
| s2t_transformer_l | 268M | 3.3 | 7.7 | 3.5 | 7.8 |
| s2t_transformer_lp | 318M | 3.1 | 7.5 | 3.4 | 7.6 |


## Example 2: Speech Translation (ST) on MuST-C

#### Data Preparation
[Download](https://ict.fbk.eu/must-c) and unpack [MuST-C](https://www.aclweb.org/anthology/N19-1202) data
to a path `${MUSTC_ROOT}/en-${TARGET_LANG_ID}`, then preprocess it with
```bash
# Generate TSV manifests, features, vocabulary and configuration for each language
python examples/speech_to_text/prep_mustc_data.py --data-root ${MUSTC_ROOT} --task asr \
    --vocab-type unigram --vocab-size 5000
python examples/speech_to_text/prep_mustc_data.py --data-root ${MUSTC_ROOT} --task st \
    --vocab-type unigram --vocab-size 8000

# Add vocabulary and configuration for joint data (based on the manifests and features generated above)
python examples/speech_to_text/prep_mustc_data.py --data-root ${MUSTC_ROOT} --task asr --joint \
    --vocab-type unigram --vocab-size 10000
python examples/speech_to_text/prep_mustc_data.py --data-root ${MUSTC_ROOT} --task st --joint \
    --vocab-type unigram --vocab-size 10000
```
The generated files will be available under `${MUSTC_ROOT}/en-${TARGET_LANG_ID}` (per-language data) and
`MUSTC_ROOT` (joint data).

#### ASR
###### Training
ASR data from En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de --train-subset train_asr --valid-subset dev_asr --save-dir ${ASR_SAVE_DIR} \
    --num-workers 4 --max-tokens 40000 --task speech_to_text --criterion label_smoothed_cross_entropy \
    --report-accuracy --max-update 100000 --arch s2t_transformer_s --optimizer adam --lr 1e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
Using joint data from all directions:
```bash
fairseq-train ${MUSTC_ROOT} \
    --train-subset train_de_asr,train_nl_asr,train_es_asr,train_fr_asr,train_it_asr,train_pt_asr,train_ro_asr,train_ru_asr \
    --valid-subset dev_de_asr,dev_nl_asr,dev_es_asr,dev_fr_asr,dev_it_asr,dev_pt_asr,dev_ro_asr,dev_ru_asr \
    --save-dir ${JOINT_ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --task speech_to_text --arch s2t_transformer_s \
    --criterion label_smoothed_cross_entropy --report-accuracy --max-update 100000 --optimizer adam --lr 1e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
where `ASR_SAVE_DIR` (`JOINT_ASR_SAVE_DIR`) is the checkpoint root path. We set `--update-freq 8` to simulate 8 GPUs
with 1 GPU. You may want to update it accordingly when using more than 1 GPU.

###### Inference & Evaluation
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
    --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de --gen-subset tst-COMMON_asr --task speech_to_text \
    --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
    --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct

# For models trained on joint data
python scripts/average_checkpoints.py --inputs ${JOINT_ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
    --output "${JOINT_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
for LANG in de nl es fr it pt ro ru; do
    fairseq-generate ${MUSTC_ROOT} --gen-subset tst-COMMON_${LANG}_asr --task speech_to_text \
        --path ${JOINT_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
        --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
done
```
###### Result
| Data | --arch | Params | En-De | En-Nl | En-Es | En-Fr | En-It | En-Pt | En-Ro | En-Ru |
|---|---|---|---|---|---|---|---|---|---|---|
| Single | s2t_transformer_s | 31M | 18.2 | 17.6 | 17.7 | 17.2 | 17.9 | 19.1 | 18.1 | 17.7 |
| Joint | s2t_transformer_m | 76M | 16.8 | 16.7 | 16.9 | 16.9 | 17.0 | 17.4 | 17.0 | 16.9 |

#### ST
###### Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de --train-subset train_st --valid-subset dev_st --save-dir ${ST_SAVE_DIR} \
    --num-workers 4 --max-tokens 40000 --task speech_to_text --criterion label_smoothed_cross_entropy \
    --report-accuracy --max-update 100000 --arch s2t_transformer_s --optimizer adam --lr 2e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
    --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```
Example for multilingual models:
```bash
fairseq-train ${MUSTC_ROOT} \
    --train-subset train_de_st,train_nl_st,train_es_st,train_fr_st,train_it_st,train_pt_st,train_ro_st,train_ru_st \
    --valid-subset dev_de_st,dev_nl_st,dev_es_st,dev_fr_st,dev_it_st,dev_pt_st,dev_ro_st,dev_ru_st \
    --save-dir ${MULTILINGUAL_ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --task speech_to_text \
    --arch s2t_transformer_s --criterion label_smoothed_cross_entropy --report-accuracy --ignore-prefix-size 1 \
    --max-update 100000 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 \
    --seed 1 --update-freq 8 --load-pretrained-encoder-from ${JOINT_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```
where `ST_SAVE_DIR` (`MULTILINGUAL_ST_SAVE_DIR`) is the checkpoint root path. The ST encoder is pre-trained by ASR
for faster training and better performance: `--load-pretrained-encoder-from <(JOINT_)ASR checkpoint path>`. We set
`--update-freq 8` to simulate 8 GPUs with 1 GPU. You may want to update it accordingly when using more than 1 GPU.
For multilingual models, we prepend target language ID token as target BOS, which should be excluded from
the training loss via `--ignore-prefix-size 1`.

###### Inference & Evaluation
Average the last 10 checkpoints and evaluate on the `tst-COMMON` split:
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
    --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT} --gen-subset tst-COMMON_st --task speech_to_text \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 --scoring sacrebleu

# For multilingual models
python scripts/average_checkpoints.py --inputs ${MULTILINGUAL_ST_SAVE_DIR} --num-epoch-checkpoints 10 \
    --output "${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
for LANG in de nl es fr it pt ro ru; do
    fairseq-generate ${MUSTC_ROOT} --gen-subset tst-COMMON_${LANG}_st --task speech_to_text --prefix-size 1 \
        --path ${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 --scoring sacrebleu
done
```
For multilingual models, we force decoding from the target language ID token (as BOS) via `--prefix-size 1`.

###### Result
| Data | --arch | Params | En-De | En-Nl | En-Es | En-Fr | En-It | En-Pt | En-Ro | En-Ru |
|---|---|---|---|---|---|---|---|---|---|---|
| Bilingual | s2t_transformer_s | 31M | 22.7 | 27.3 | 27.2 | 32.9 | 22.7 | 28.1 | 21.9 | 15.3 |
| Multilingual | s2t_transformer_m | 76M | 24.5 | 28.6 | 28.2 | 34.9 | 24.6 | 31.1 | 23.8 | 16.0 |


## Example 3: ST on CoVoST
We replicate the experiments in
[CoVoST 2 and Massively Multilingual Speech-to-Text Translation (Wang et al., 2020)](https://arxiv.org/abs/2007.10310).

#### Data Preparation
Download and preprocess [CoVoST (version 2)](https://arxiv.org/abs/2007.10310) data with
```bash
# En ASR
python examples/speech_to_text/prep_covost_data.py --data-root ${COVOST_ROOT} --vocab-type char --src-lang en
# ST
python examples/speech_to_text/prep_covost_data.py --data-root ${COVOST_ROOT} --vocab-type char \
    --src-lang fr --tgt-lang en
```
where `COVOST_ROOT` is the root path for downloaded data as well as generated manifest and feature files.

#### ASR
###### Training
```bash
fairseq-train ${COVOST_ROOT} --train-subset train_asr --valid-subset dev_asr --save-dir ${ASR_SAVE_DIR} \
    --num-workers 4 --max-tokens 40000 --task speech_to_text --criterion label_smoothed_cross_entropy \
    --report-accuracy --max-update 100000 --arch s2t_transformer_s --optimizer adam --lr 1e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
where `ASR_SAVE_DIR` is the checkpoint root path. We set `--update-freq 8` to simulate 8 GPUs with 1 GPU.
You may want to update it accordingly when using more than 1 GPU.

###### Inference & Evaluation
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
    --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${COVOST_ROOT} --gen-subset test_asr_en --task speech_to_text \
    --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
    --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
```
###### Result
| --arch | Params | En |
|---|---|---|
| s2t_transformer_s | 31M | 25.6 |

#### ST
###### Training
```bash
fairseq-train ${COVOST_ROOT} --train-subset train_st_fr_en --valid-subset dev_st_fr_en --save-dir ${ST_SAVE_DIR} \
    --num-workers 4 --max-tokens 40000 --task speech_to_text --criterion label_smoothed_cross_entropy \
    --report-accuracy --max-update 100000 --arch s2t_transformer_s --optimizer adam --lr 2e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
    --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```
where `ST_SAVE_DIR` is the checkpoint root path. The ST encoder is pre-trained by En ASR for faster training and better
performance: `--load-pretrained-encoder-from <ASR checkpoint path>`. We set `--update-freq 8` to simulate 8 GPUs with 1 GPU.
You may want to update it accordingly when using more than 1 GPU.

###### Inference & Evaluation
Average the last 10 checkpoints and evaluate on test split:
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
    --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${COVOST_ROOT} --gen-subset test_st_fr_en --task speech_to_text \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 --scoring sacrebleu
```

###### Result
| --arch | Params | Fr-En | De-En | Es-En | Ca-En | En-De | En-Ca | En-Fa | En-Et |
|---|---|---|---|---|---|---|---|---|---|
| s2t_transformer_s | 31M | 26.3 | 17.1 | 23.0 | 18.8 | 16.3 | 21.8 | 13.1 | 13.2 |

## Citation
Please cite as:
```
@inproceedings{wang2020fairseqs2t,
  title = {fairseq S2T: Fast Speech-to-Text Modeling with fairseq},
  author = {Changhan Wang and Yun Tang and Xutai Ma and Anne Wu and Dmytro Okhonko and Juan Pino},
  booktitle = {Proceedings of the 2020 Conference of the Asian Chapter of the Association for Computational Linguistics (AACL): System Demonstrations},
  year = {2020},
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```

## More Paper Code
The following papers also base their experiments on fairseq S2T. We are adding more examples for replication.

- [Improving Cross-Lingual Transfer Learning for End-to-End Speech Recognition with Speech Translation (Wang et al., 2020)](https://arxiv.org/abs/2006.05474)
- [Self-Supervised Representations Improve End-to-End Speech Translation (Wu et al., 2020)](https://arxiv.org/abs/2006.12124)
- [Self-Training for End-to-End Speech Translation (Pino et al., 2020)](https://arxiv.org/abs/2006.02490)
- [CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus (Wang et al., 2020)](https://arxiv.org/abs/2002.01320)
- [Harnessing Indirect Training Data for End-to-End Automatic Speech Translation: Tricks of the Trade (Pino et al., 2019)](https://arxiv.org/abs/1909.06515)
