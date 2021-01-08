[[Back]](..)

# S2T Example: Speech Translation (ST) on MuST-C

[MuST-C](https://www.aclweb.org/anthology/N19-1202) is multilingual speech-to-text translation corpus with
8-language translations on English TED talks. We match the state-of-the-art performance in
[ESPNet-ST](https://arxiv.org/pdf/2004.10234.pdf) with a simpler model training pipeline.

## Data Preparation
[Download](https://ict.fbk.eu/must-c) and unpack MuST-C data to a path
`${MUSTC_ROOT}/en-${TARGET_LANG_ID}`, then preprocess it with
```bash
# additional Python packages for S2T data processing/model training
pip install pandas torchaudio sentencepiece

# Generate TSV manifests, features, vocabulary
# and configuration for each language
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 5000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 8000

# Add vocabulary and configuration for joint data
# (based on the manifests and features generated above)
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr --joint \
  --vocab-type unigram --vocab-size 10000
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st --joint \
  --vocab-type unigram --vocab-size 10000
```
The generated files (manifest, features, vocabulary and data configuration) will be added to
`${MUSTC_ROOT}/en-${TARGET_LANG_ID}` (per-language data) and `MUSTC_ROOT` (joint data).

Download our vocabulary files if you want to use our pre-trained models:
- ASR: [En-De](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_de_asr_vocab_unigram5000.zip), [En-Nl](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_nl_asr_vocab_unigram5000.zip), [En-Es](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_es_asr_vocab_unigram5000.zip), [En-Fr](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_fr_asr_vocab_unigram5000.zip), [En-It](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_it_asr_vocab_unigram5000.zip), [En-Pt](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_pt_asr_vocab_unigram5000.zip), [En-Ro](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_ro_asr_vocab_unigram5000.zip), [En-Ru](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_ru_asr_vocab_unigram5000.zip), [Joint](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_vocab_unigram10000.zip)
- ST: [En-De](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_de_st_vocab_unigram8000.zip), [En-Nl](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_nl_st_vocab_unigram8000.zip), [En-Es](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_es_st_vocab_unigram8000.zip), [En-Fr](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_fr_st_vocab_unigram8000.zip), [En-It](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_it_st_vocab_unigram8000.zip), [En-Pt](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_pt_st_vocab_unigram8000.zip), [En-Ro](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_ro_st_vocab_unigram8000.zip), [En-Ru](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_ru_st_vocab_unigram8000.zip), [Multilingual](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_multilingual_st_vocab_unigram10000.zip)

## ASR
#### Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
For joint model (using ASR data from all 8 directions):
```bash
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_asr.yaml \
  --train-subset train_de_asr,train_nl_asr,train_es_asr,train_fr_asr,train_it_asr,train_pt_asr,train_ro_asr,train_ru_asr \
  --valid-subset dev_de_asr,dev_nl_asr,dev_es_asr,dev_fr_asr,dev_it_asr,dev_pt_asr,dev_ro_asr,dev_ru_asr \
  --save-dir ${JOINT_ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
where `ASR_SAVE_DIR` (`JOINT_ASR_SAVE_DIR`) is the checkpoint root path. We set `--update-freq 8` to simulate 8 GPUs
with 1 GPU. You may want to update it accordingly when using more than 1 GPU.

#### Inference & Evaluation
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --gen-subset tst-COMMON_asr --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct

# For models trained on joint data
python scripts/average_checkpoints.py \
  --inputs ${JOINT_ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${JOINT_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
for LANG in de nl es fr it pt ro ru; do
  fairseq-generate ${MUSTC_ROOT} \
  --config-yaml config_asr.yaml --gen-subset tst-COMMON_${LANG}_asr --task speech_to_text \
    --path ${JOINT_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
    --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct
done
```
#### Results
| Data | --arch | Params | En-De | En-Nl | En-Es | En-Fr | En-It | En-Pt | En-Ro | En-Ru | Model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Single | s2t_transformer_s | 31M | [18.2](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_de_asr_transformer_s.pt) | [17.6](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_nl_asr_transformer_s.pt) | [17.7](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_es_asr_transformer_s.pt) | [17.2](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_fr_asr_transformer_s.pt) | [17.9](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_it_asr_transformer_s.pt) | [19.1](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_pt_asr_transformer_s.pt) | [18.1](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_ro_asr_transformer_s.pt) | [17.7](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_ru_asr_transformer_s.pt) | (<-Download) |
| Joint | s2t_transformer_m | 76M | 16.8 | 16.7 | 16.9 | 16.9 | 17.0 | 17.4 | 17.0 | 16.9 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_joint_asr_transformer_m.pt) |

## ST
#### Training
En-De as example:
```bash
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```
For multilingual model (all 8 directions):
```bash
fairseq-train ${MUSTC_ROOT} \
  --config-yaml config_st.yaml \
  --train-subset train_de_st,train_nl_st,train_es_st,train_fr_st,train_it_st,train_pt_st,train_ro_st,train_ru_st \
  --valid-subset dev_de_st,dev_nl_st,dev_es_st,dev_fr_st,dev_it_st,dev_pt_st,dev_ro_st,dev_ru_st \
  --save-dir ${MULTILINGUAL_ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --ignore-prefix-size 1 --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${JOINT_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}
```
where `ST_SAVE_DIR` (`MULTILINGUAL_ST_SAVE_DIR`) is the checkpoint root path. The ST encoder is pre-trained by ASR
for faster training and better performance: `--load-pretrained-encoder-from <(JOINT_)ASR checkpoint path>`. We set
`--update-freq 8` to simulate 8 GPUs with 1 GPU. You may want to update it accordingly when using more than 1 GPU.
For multilingual models, we prepend target language ID token as target BOS, which should be excluded from
the training loss via `--ignore-prefix-size 1`.

#### Inference & Evaluation
Average the last 10 checkpoints and evaluate on the `tst-COMMON` split:
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${MUSTC_ROOT}/en-de \
  --config-yaml config_st.yaml --gen-subset tst-COMMON_st --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu

# For multilingual models
python scripts/average_checkpoints.py \
  --inputs ${MULTILINGUAL_ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
for LANG in de nl es fr it pt ro ru; do
  fairseq-generate ${MUSTC_ROOT} \
    --config-yaml config_st.yaml --gen-subset tst-COMMON_${LANG}_st --task speech_to_text \
    --prefix-size 1 --path ${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring sacrebleu
done
```
For multilingual models, we force decoding from the target language ID token (as BOS) via `--prefix-size 1`.

#### Results
| Data | --arch | Params | En-De | En-Nl | En-Es | En-Fr | En-It | En-Pt | En-Ro | En-Ru | Model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bilingual | s2t_transformer_s | 31M | [22.7](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_de_st_transformer_s.pt) | [27.3](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_nl_st_transformer_s.pt) | [27.2](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_es_st_transformer_s.pt) | [32.9](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_fr_st_transformer_s.pt) | [22.7](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_it_st_transformer_s.pt) | [28.1](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_pt_st_transformer_s.pt) | [21.9](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_ro_st_transformer_s.pt) | [15.3](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_ru_st_transformer_s.pt) | (<-Download) |
| Multilingual | s2t_transformer_m | 76M | 24.5 | 28.6 | 28.2 | 34.9 | 24.6 | 31.1 | 23.8 | 16.0 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/mustc_multilingual_st_transformer_m.pt) |

[[Back]](..)
