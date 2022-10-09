# Unit to Speech Model (unit2speech)

Unit to speech model is modified Tacotron2 model that learns to synthesize speech from discrete speech units. All models are trained on quantized [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

Upstream Units | Download Links | model md5
|-|-|-
Log Mel Filterbank + KM50 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km50/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km50/code_dict) | 932b3b8527c0125f5f964b57762eba49
Log Mel Filterbank + KM100 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km100/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km100/code_dict) | cde0b0d278a39011d0acbd5df27abdf4
Log Mel Filterbank + KM200 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km200/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/tts_km200/code_dict) | dba0f1d4de64bc7976718834010b23e7
Modified CPC + KM50 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km50/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km50/code_dict) | a585e8dd8890ea56164f17635dd8e613
Modified CPC + KM100 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km100/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km100/code_dict) | 5c0ee2869b4f483d17f37f1a41a548e0
Modified CPC + KM200 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km200/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/tts_km200/code_dict) | 2f0c9951cf37020d9464514bff48bc5d
HuBERT Base + KM50 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km50/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km50/code_dict) | 85ffce8baec5aa90035ab696fe676fce
HuBERT Base + KM100 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km100/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km100/code_dict) | df4a9c6ffd1bb00c91405432c234aba3
HuBERT Base + KM200 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km200/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/tts_km200/code_dict) | ac72f2c0c563589819bec116c7f8d274
wav2vec 2.0 Large + KM50 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km50/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km50/code_dict) | e3503d0ad822b2c24b89f68b857fedff
wav2vec 2.0 Large + KM100 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km100/tts_checkpoint_best.pt) - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km100/code_dict) | eb3666e456ae4c96bf2a1eec825c13ed
wav2vec 2.0 Large + KM200 | [model](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km200/tts_checkpoint_best.pt)  - [code_dict](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/tts_km200/code_dict) | 777d343e963c4d64f04d78eef032f4e8

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
CODE_DICT_PATH=<unit2speech_code_dict_path>

PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech/synthesize_audio_from_units.py \
    --tts_model_path $TTS_MODEL_PATH \
    --quantized_unit_path $QUANTIZED_UNIT_PATH \
    --out_audio_dir $OUT_DIR \
    --waveglow_path  $WAVEGLOW_PATH \
    --code_dict_path $CODE_DICT_PATH \
    --max_decoder_steps 2000
```
