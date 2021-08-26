# Unit to Speech Model (unit2speech)

Unit to speech model is modified Tacotron2 model that learns to synthesize speech from discrete speech units. All models are trained on quantized [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

Upstream Units | Download Link
|-|-
Log Mel Filterbank + KM50 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km50/tts_checkpoint_best.pt)
Log Mel Filterbank + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km100/tts_checkpoint_best.pt)
Log Mel Filterbank + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km200/tts_checkpoint_best.pt)
Log Mel Filterbank + KM500 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km500/tts_checkpoint_best.pt)
Modified CPC + KM50 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km50/tts_checkpoint_best.pt)
Modified CPC + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km100/tts_checkpoint_best.pt)
Modified CPC + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km200/tts_checkpoint_best.pt)
Modified CPC + KM500 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km500/tts_checkpoint_best.pt)
HuBERT Base + KM50 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km50/tts_checkpoint_best.pt)
HuBERT Base + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km100/tts_checkpoint_best.pt)
HuBERT Base + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km200/tts_checkpoint_best.pt)
HuBERT Base + KM500 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km500/tts_checkpoint_best.pt)
wav2vec 2.0 Large + KM50 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km50/tts_checkpoint_best.pt)
wav2vec 2.0 Large + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km100/tts_checkpoint_best.pt)
wav2vec 2.0 Large + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km200/tts_checkpoint_best.pt)
wav2vec 2.0 Large + KM500 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km500/tts_checkpoint_best.pt)

## Run inference using a unit2speech model
* Install librosa, unidecode and inflect using `pip install librosa, unidecode, inflect`
* Download [Waveglow checkpoint](https://dl.fbaipublicfiles.com/textless_nlp/gslm/waveglow_256channels_new.pt). This is the vocoder.

Sample commnd to run inference using trained unit2speech models. Please note that the quantized audio to synthesized should be using the same units as the unit2speech model was trained with.
```
FAIRSEQ_ROOT=<path_to_your_fairseq_repo_root>
TTS_MODEL_PATH=<unit2speech_model_file_path>
QUANTIZED_UNIT_PATH=<quantized_audio_file_path>
OUT_DIR=<dir_to_dump_synthesized_audio_files>
WAVEGLOW_PATH=<path_where_you_have_downloaded_waveglow_checkpoint>

PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech/synthesize_audio_from_units.py \
    --tts_model_path $TTS_MODEL_PATH \
    --quantized_unit_path $QUANTIZED_UNIT_PATH \
    --out_audio_dir $OUT_DIR \
    --waveglow_path  $WAVEGLOW_PATH \
    --max_decoder_steps 2000
```