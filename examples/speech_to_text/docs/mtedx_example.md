[[Back]](..)

# S2T Example: Speech Translation (ST) on Multilingual TEDx

[Multilingual TEDx](https://arxiv.org/abs/2102.01757) is multilingual corpus for speech recognition and
speech translation. The data is derived from TEDx talks in 8 source languages
with translations to a subset of 5 target languages.

## Data Preparation
[Download](http://openslr.org/100/) and unpack Multilingual TEDx data to a path
`${MTEDX_ROOT}/${LANG_PAIR}`, then preprocess it with
```bash
# additional Python packages for S2T data processing/model training
pip install pandas torchaudio sentencepiece

# Generate TSV manifests, features, vocabulary
# and configuration for each language
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task asr \
  --vocab-type unigram --vocab-size 1000
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task st \
  --vocab-type unigram --vocab-size 1000

# Add vocabulary and configuration for joint data
# (based on the manifests and features generated above)
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task asr --joint \
  --vocab-type unigram --vocab-size 8000
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task st --joint \
  --vocab-type unigram --vocab-size 8000
```
The generated files (manifest, features, vocabulary and data configuration) will be added to
`${MTEDX_ROOT}/${LANG_PAIR}` (per-language data) and `MTEDX_ROOT` (joint data).


## ASR
#### Training
Spanish as example:
```bash
fairseq-train ${MTEDX_ROOT}/es-es \
    --config-yaml config_asr.yaml --train-subset train_asr --valid-subset valid_asr \
    --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-epoch 200 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch s2t_transformer_xs --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --dropout 0.3 --label-smoothing 0.1 \
    --load-pretrained-encoder-from ${PRETRAINED_ENCODER} \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 10 --update-freq 8 --patience 10
```
For joint model (using ASR data from all 8 languages):
```bash
fairseq-train ${MTEDX_ROOT} \
    --config-yaml config_asr.yaml \
    --train-subset train_es-es_asr,train_fr-fr_asr,train_pt-pt_asr,train_it-it_asr,train_ru-ru_asr,train_el-el_asr,train_ar-ar_asr,train_de-de_asr \
    --valid-subset valid_es-es_asr,valid_fr-fr_asr,valid_pt-pt_asr,valid_it-it_asr,valid_ru-ru_asr,valid_el-el_asr,valid_ar-ar_asr,valid_de-de_asr \
    --save-dir ${MULTILINGUAL_ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-epoch 200 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --dropout 0.3 --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 10 --update-freq 8 --patience 10 \
    --ignore-prefix-size 1
```
where `MULTILINGUAL_ASR_SAVE_DIR` is the checkpoint root path. We set `--update-freq 8` to simulate 8 GPUs
with 1 GPU. You may want to update it accordingly when using more than 1 GPU.
For multilingual models, we prepend target language ID token as target BOS, which should be excluded from
the training loss via `--ignore-prefix-size 1`.

#### Inference & Evaluation
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${MTEDX_ROOT}/es-es \
  --config-yaml config_asr.yaml --gen-subset test --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --skip-invalid-size-inputs-valid-test \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct --remove-bpe

# For models trained on joint data
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${MULTILINGUAL_ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${MULTILINGUAL_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"

for LANG in es fr pt it ru el ar de; do
  fairseq-generate ${MTEDX_ROOT} \
    --config-yaml config_asr.yaml --gen-subset test_${LANG}-${LANG}_asr --task speech_to_text \
    --prefix-size 1 --path ${MULTILINGUAL_ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 40000 --beam 5 \
    --skip-invalid-size-inputs-valid-test \
    --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct --remove-bpe
done
```
#### Results
| Data         | --arch             | Params |  Es  |  Fr  |  Pt  |  It  |  Ru  |   El  |   Ar  |   De  |
|--------------|--------------------|--------|------|------|------|------|------|-------|-------|-------|
| Monolingual  | s2t_transformer_xs |    10M | 46.4 | 45.6 | 54.8 | 48.0 | 74.7 | 109.5 | 104.4 | 111.1 |


## ST
#### Training
Es-En as example:
```bash
fairseq-train ${MTEDX_ROOT}/es-en \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset valid_st \
    --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-epoch 200 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch s2t_transformer_xs --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --dropout 0.3 --label-smoothing 0.1 \
    --load-pretrained-encoder-from ${PRETRAINED_ENCODER} \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 10 --update-freq 8 --patience 10
```
For multilingual model (all 12 directions):
```bash
fairseq-train ${MTEDX_ROOT} \
    --config-yaml config_st.yaml \
    --train-subset train_el-en_st,train_es-en_st,train_es-fr_st,train_es-it_st,train_es-pt_st,train_fr-en_st,train_fr-es_st,train_fr-pt_st,train_it-en_st,train_it-es_st,train_pt-en_st,train_pt-es_st,train_ru-en_st \
    --valid-subset valid_el-en_st,valid_es-en_st,valid_es-fr_st,valid_es-it_st,valid_es-pt_st,valid_fr-en_st,valid_fr-es_st,valid_fr-pt_st,valid_it-en_st,valid_it-es_st,valid_pt-en_st,valid_pt-es_st,valid_ru-en_st \
    --save-dir ${MULTILINGUAL_ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-epoch 200 \
    --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 --clip-norm 10.0 --seed 1 --dropout 0.3 --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --keep-last-epochs 10 --update-freq 8 --patience 10 \
    --ignore-prefix-size 1 \
    --load-pretrained-encoder-from ${PRETRAINED_ENCODER}
```
where `ST_SAVE_DIR` (`MULTILINGUAL_ST_SAVE_DIR`) is the checkpoint root path. The ST encoder is pre-trained by ASR
for faster training and better performance: `--load-pretrained-encoder-from <(JOINT_)ASR checkpoint path>`. We set
`--update-freq 8` to simulate 8 GPUs with 1 GPU. You may want to update it accordingly when using more than 1 GPU.
For multilingual models, we prepend target language ID token as target BOS, which should be excluded from
the training loss via `--ignore-prefix-size 1`.

#### Inference & Evaluation
Average the last 10 checkpoints and evaluate on the `test` split:
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${MTEDX_ROOT}/es-en \
  --config-yaml config_st.yaml --gen-subset test --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu --remove-bpe

# For multilingual models
python scripts/average_checkpoints.py \
  --inputs ${MULTILINGUAL_ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

for LANGPAIR in es-en es-fr es-pt fr-en fr-es fr-pt pt-en pt-es it-en it-es ru-en el-en; do
  fairseq-generate ${MTEDX_ROOT} \
    --config-yaml config_st.yaml --gen-subset test_${LANGPAIR}_st --task speech_to_text \
    --prefix-size 1 --path ${MULTILINGUAL_ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 40000 --beam 5 \
    --skip-invalid-size-inputs-valid-test \
    --scoring sacrebleu --remove-bpe
done
```
For multilingual models, we force decoding from the target language ID token (as BOS) via `--prefix-size 1`.

#### Results
| Data         | --arch          | Params | Es-En | Es-Pt | Es-Fr | Fr-En | Fr-Es | Fr-Pt | Pt-En | Pt-Es | It-En | It-Es | Ru-En | El-En |
|--------------|--------------------|-----|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Bilingual    | s2t_transformer_xs | 10M |  7.0  |  12.2 |  1.7  |  8.9  |  10.6 |  7.9  |  8.1  |  8.7  |   6.4 |  1.0  |  0.7  |  0.6  |
| Multilingual | s2t_transformer_s  | 31M |  12.3 |  17.4 |   6.1 |  12.0 |  13.6 |  13.2 |  12.0 |  13.7 |  10.7 |  13.1 |  0.6  |  0.8  |


## Citation
Please cite as:
```
@misc{salesky2021mtedx,
      title={Multilingual TEDx Corpus for Speech Recognition and Translation},
      author={Elizabeth Salesky and Matthew Wiesner and Jacob Bremerman and Roldano Cattoni and Matteo Negri and Marco Turchi and Douglas W. Oard and Matt Post},
      year={2021},
}

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

[[Back]](..)
