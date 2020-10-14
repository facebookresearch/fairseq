# Speech-to-Text (S2T) Modeling

## Data Preparation
S2T modeling data consists of source speech features, target text and other optional information
(source text, speaker id, etc.). Fairseq S2T uses per-dataset-split TSV manifest files
to store these information. Each data field is represented by a column in the TSV file.

Unlike text token embeddings, speech features (e.g. log mel-filter banks) are usually fixed
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
Download and preprocess LibriSpeech data with
```bash
python examples/speech_to_text/prep_librispeech_data.py \
    --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
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
python scripts/average_checkpoints.py \
    --inputs ${SAVE_DIR} --num-epoch-checkpoints 10 --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
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
[Download](https://ict.fbk.eu/must-c) and unpack MuST-C data to a path `MUSTC_ROOT`, then preprocess it with
```bash
python examples/speech_to_text/prep_mustc_data.py --data-root ${MUSTC_ROOT} \
    --asr-vocab-type unigram --asr-vocab-size 5000 \
    --st-vocab-type unigram --st-vocab-size 8000
```
The generated manifest and feature files will be available under `MUSTC_ROOT`.

#### ASR
###### Training
```bash
fairseq-train ${MUSTC_ROOT} --train-subset train_asr --valid-subset dev_asr --save-dir ${ASR_SAVE_DIR} \
    --num-workers 4 --max-tokens 40000 --task speech_to_text --criterion label_smoothed_cross_entropy \
    --report-accuracy --max-update 100000 --arch s2t_transformer_s --optimizer adam --lr 1e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
where `ASR_SAVE_DIR` is the checkpoint root path. We set `--update-freq 8` to simulate 8 GPUs with 1 GPU.
You may want to update it accordingly when using more than 1 GPU.

###### Inference & Evaluation
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
    --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT} --gen-subset tst-COMMON_asr --task speech_to_text \
    --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
    --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
```
###### Result
| --arch | Params | En-De | En-Nl | En-Es | En-Fr | En-It | En-Pt | En-Ro | En-Ru |
|---|---|---|---|---|---|---|---|---|---|
| s2t_transformer_s | 31M | 18.2 | 17.6 | 17.7 | 17.2 | 17.9 | 19.1 | 18.1 | 17.7 |

#### ST
###### Training
```bash
fairseq-train ${MUSTC_ROOT} --train-subset train_st --valid-subset dev_st --save-dir ${ST_SAVE_DIR} \
    --num-workers 4 --max-tokens 40000 --task speech_to_text --criterion label_smoothed_cross_entropy \
    --report-accuracy --max-update 100000 --arch s2t_transformer_s --optimizer adam --lr 2e-3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
    --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```
where `ST_SAVE_DIR` is the checkpoint root path. The ST encoder is pre-trained by ASR for faster training and better
performance: `--load-pretrained-encoder-from <ASR checkpoint path>`. We set `--update-freq 8` to simulate 8 GPUs with 1 GPU.
You may want to update it accordingly when using more than 1 GPU.

###### Inference & Evaluation
Average the last 10 checkpoints and evaluate on the `tst-COMMON` split:
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
    --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT} --gen-subset tst-COMMON_st --task speech_to_text \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 --scoring sacrebleu
```

###### Result
| --arch | Params | En-De | En-Nl | En-Es | En-Fr | En-It | En-Pt | En-Ro | En-Ru |
|---|---|---|---|---|---|---|---|---|---|
| s2t_transformer_s | 31M | 22.7 | 27.3 | 27.2 | 32.9 | 22.7 | 28.1 | 21.9 | 15.3 |


## Example 3: ST on CoVoST
#### Data Preparation
Download and preprocess CoVoST data with
```bash
# En ASR
python examples/speech_to_text/prep_covost_data.py --data-root ${COVOST_ROOT} \
    --vocab-type char --src-lang en
# ST
python examples/speech_to_text/prep_covost_data.py --data-root ${COVOST_ROOT} \
    --vocab-type char --src-lang fr --tgt-lang en
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
python scripts/average_checkpoints.py \
    --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
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
python scripts/average_checkpoints.py \
    --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
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
