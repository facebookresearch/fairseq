# GSLM Tools

## Resynthesis
You can use the command line tool below to input an audio file and get the resynthesized audio. This tool implements the unsupervised method for resynthesis described in the paper. The way to invoke the command line tool is shown below.
```
FAIRSEQ_ROOT=<path_to_your_fairseq_repo_root>
TYPE=<one_of_logmel/cpc/hubert/w2v2>
ACOUSTIC_MODEL_PATH=<path_of_pretrained_acoustic_model>
LAYER=<layer_of_acoustic_model_to_extract_features_from>
KM_MODEL_PATH=<output_path_of_the_kmeans_model>
TTS_MODEL_PATH=<unit2speech_model_file_path>
WAVEGLOW_PATH=<path_where_you_have_downloaded_waveglow_checkpoint>

PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/tools/gen_speech.py \
    --feature_type $TYPE \
    --acoustic_model_path $ACOUSTIC_MODEL_PATH \
    --layer $LAYER \
    --kmeans_model_path $KM_MODEL_PATH \
    --tts_model_path $TTS_MODEL_PATH \
    --waveglow_path  $WAVEGLOW_PATH \
    --max_decoder_steps 2000
```