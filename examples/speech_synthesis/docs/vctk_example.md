[[Back]](..)

# VCTK

[VCTK](https://datashare.ed.ac.uk/handle/10283/3443) is an open English speech corpus. We provide examples
for building [Transformer](https://arxiv.org/abs/1809.08895) models on this dataset.


## Data preparation
Download data, create splits and generate audio manifests with
```bash
python -m examples.speech_synthesis.preprocessing.get_vctk_audio_manifest \
  --output-data-root ${AUDIO_DATA_ROOT} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT}
```

To denoise audio and trim leading/trailing silence using signal processing based VAD, run
```bash
for SPLIT in dev test train; do
    python -m examples.speech_synthesis.preprocessing.denoise_and_vad_audio \
      --audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
      --output-dir ${PROCESSED_DATA_ROOT} \
      --denoise --vad --vad-agg-level 3
done
```
which generates a new audio TSV manifest under `${PROCESSED_DATA_ROOT}` with updated path to the processed audio and
a new column for SNR.

To do filtering by CER, follow the [Automatic Evaluation](../docs/ljspeech_example.md#automatic-evaluation) section to
run ASR model (add `--eval-target` to `get_eval_manifest` for evaluation on the reference audio; add `--err-unit char`
to `eval_asr` to compute CER instead of WER). The example-level CER is saved to
`${EVAL_OUTPUT_ROOT}/uer_cer.${SPLIT}.tsv`.

Then, extract log-Mel spectrograms, generate feature manifest and create data configuration YAML with
```bash
python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${PROCESSED_DATA_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT} \
  --ipa-vocab --use-g2p \
  --snr-threshold 15 \
  --cer-threshold 0.1 --cer-tsv-path ${EVAL_OUTPUT_ROOT}/uer_cer.${SPLIT}.tsv
```
where we use phoneme inputs (`--ipa-vocab --use-g2p`) as example. For sample filtering, we set the SNR and CER threshold
to 15 and 10%, respectively.

## Training
(Please refer to [the LJSpeech example](../docs/ljspeech_example.md#transformer).)

## Inference
(Please refer to [the LJSpeech example](../docs/ljspeech_example.md#inference).)

## Automatic Evaluation
(Please refer to [the LJSpeech example](../docs/ljspeech_example.md#automatic-evaluation).)

## Results

| --arch | Params | Test MCD | Model |
|---|---|---|---|
| tts_transformer | 54M | 3.4 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2/vctk_transformer_phn.tar) |

[[Back]](..)
