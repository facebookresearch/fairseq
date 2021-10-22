[[Back]](..)

# LJSpeech

[LJSpeech](https://keithito.com/LJ-Speech-Dataset) is a public domain TTS
corpus with around 24 hours of English speech sampled at 22.05kHz. We provide examples for building
[Transformer](https://arxiv.org/abs/1809.08895) and [FastSpeech 2](https://arxiv.org/abs/2006.04558)
models on this dataset.


## Data preparation

Download data, create splits and generate audio manifests with
```bash
python -m examples.speech_synthesis.preprocessing.get_ljspeech_audio_manifest \
  --output-data-root ${AUDIO_DATA_ROOT} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT}
```

Then, extract log-Mel spectrograms, generate feature manifest and create data configuration YAML with
```bash
python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT} \
  --ipa-vocab --use-g2p
```
where we use phoneme inputs (`--ipa-vocab --use-g2p`) as example.

FastSpeech 2 additionally requires frame durations, pitch and energy as auxiliary training targets.
Add `--add-fastspeech-targets` to include these fields in the feature manifests. We get frame durations either from
phoneme-level force-alignment or frame-level pseudo-text unit sequence. They should be pre-computed and specified via:
- `--textgrid-zip ${TEXT_GRID_ZIP_PATH}` for a ZIP file, inside which there is one
  [TextGrid](https://www.fon.hum.uva.nl/praat/manual/TextGrid.html) file per sample to provide force-alignment info.
- `--id-to-units-tsv ${ID_TO_UNIT_TSV}` for a TSV file, where there are 2 columns for sample ID and
  space-delimited pseudo-text unit sequence, respectively.

For your convenience, we provide pre-computed
[force-alignment](https://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_mfa.zip) from
[Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) and
[pseudo-text units](s3://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_hubert.tsv) from
[HuBERT](https://github.com/pytorch/fairseq/tree/main/examples/hubert). You can also generate them by yourself using
a different software or model.


## Training
#### Transformer
```bash
fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-tokens 30000 --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss
```
where `SAVE_DIR` is the checkpoint root path. We set `--update-freq 8` to simulate 8 GPUs with 1 GPU. You may want to
update it accordingly when using more than 1 GPU.

#### FastSpeech2
```bash
fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-sentences 6 --max-update 200000 \
  --task text_to_speech --criterion fastspeech2 --arch fastspeech2 \
  --clip-norm 5.0 --n-frames-per-step 1 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss
```


## Inference
Average the last 5 checkpoints, generate the test split spectrogram and waveform using the default Griffin-Lim vocoder:
```bash
SPLIT=test
CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 5 \
  --output ${CHECKPOINT_PATH}

python -m examples.speech_synthesis.generate_waveform ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task text_to_speech \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --dump-waveforms
```
which dumps files (waveform, feature, attention plot, etc.) to `${SAVE_DIR}/generate-${CHECKPOINT_NAME}-${SPLIT}`. To
re-synthesize target waveforms for automatic evaluation, add `--dump-target`.

## Automatic Evaluation
To start with, generate the manifest for synthetic speech, which will be taken as inputs by evaluation scripts.
```bash
python -m examples.speech_synthesis.evaluation.get_eval_manifest \
  --generation-root ${SAVE_DIR}/generate-${CHECKPOINT_NAME}-${SPLIT} \
  --audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
  --output-path ${EVAL_OUTPUT_ROOT}/eval.tsv \
  --vocoder griffin_lim --sample-rate 22050 --audio-format flac \
  --use-resynthesized-target
```
Speech recognition (ASR) models usually operate at lower sample rates (e.g. 16kHz). For the WER/CER metric,
you may need to resample the audios accordingly --- add `--output-sample-rate 16000` for `generate_waveform.py` and
use `--sample-rate 16000` for `get_eval_manifest.py`.


#### WER/CER metric
We use wav2vec 2.0 ASR model as example. [Download](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec)
the model checkpoint and dictionary, then compute WER/CER with
```bash
python -m examples.speech_synthesis.evaluation.eval_asr \
  --audio-header syn --text-header text --err-unit char --split ${SPLIT} \
  --w2v-ckpt ${WAV2VEC2_CHECKPOINT_PATH} --w2v-dict-dir ${WAV2VEC2_DICT_DIR} \
  --raw-manifest ${EVAL_OUTPUT_ROOT}/eval_16khz.tsv --asr-dir ${EVAL_OUTPUT_ROOT}/asr
```

#### MCD/MSD metric
```bash
python -m examples.speech_synthesis.evaluation.eval_sp \
  ${EVAL_OUTPUT_ROOT}/eval.tsv --mcd --msd
```

#### F0 metrics
```bash
python -m examples.speech_synthesis.evaluation.eval_f0 \
  ${EVAL_OUTPUT_ROOT}/eval.tsv --gpe --vde --ffe
```


## Results

| --arch | Params | Test MCD | Model |
|---|---|---|---|
| tts_transformer | 54M | 3.8 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_transformer_phn.tar) |
| fastspeech2 | 41M | 3.8 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_fastspeech2_phn.tar) |

[[Back]](..)
