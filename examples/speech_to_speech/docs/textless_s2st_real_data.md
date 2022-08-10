# Textless Speech-to-Speech Translation (S2ST) on Real Data

We provide instructions and pre-trained models for the work "[Textless Speech-to-Speech Translation on Real Data (Lee et al. 2021)](https://arxiv.org/abs/2112.08352)".

## Pre-trained Models

### HuBERT
Model | Pretraining Data | Model | Quantizer
|---|---|---|---
mHuBERT Base | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) En, Es, Fr speech from the 100k subset | [download](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt) | [L11 km1000](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin)


### Unit-based HiFi-GAN vocoder
Unit config | Unit size | Vocoder language | Dataset | Model
|---|---|---|---|---
mHuBERT, layer 11 | 1000 | En | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)
mHuBERT, layer 11 | 1000 | Es | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json)
mHuBERT, layer 11 | 1000 | Fr | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json)


### Speech normalizer
Language | Training data | Target unit config | Model
|---|---|---|---
En | 10 mins | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/en/en_10min.tar.gz)
En | 1 hr | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/en/en_1h.tar.gz)
En | 10 hrs | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/en/en_10h.tar.gz)
Es | 10 mins | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/es/es_10min.tar.gz)
Es | 1 hr | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/es/es_1h.tar.gz)
Es | 10 hrs | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/es/es_10h.tar.gz)
Fr | 10 mins | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/fr/fr_10min.tar.gz)
Fr | 1 hr | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/fr/fr_1h.tar.gz)
Fr | 10 hrs | mHuBERT, layer 11, km1000 | [download](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/speech_normalizer/fr/fr_10h.tar.gz)

* Refer to the paper for the details of the training data.

## Inference with Pre-trained Models

### Speech normalizer
1. Download the pre-trained models, including the dictionary, to `DATA_DIR`.
2. Format the audio data.
```bash
# AUDIO_EXT: audio extension, e.g. wav, flac, etc.
# Assume all audio files are at ${AUDIO_DIR}/*.${AUDIO_EXT}

python examples/speech_to_speech/preprocessing/prep_sn_data.py \
  --audio-dir ${AUDIO_DIR} --ext ${AUIDO_EXT} \
  --data-name ${GEN_SUBSET} --output-dir ${DATA_DIR} \
  --for-inference
```

3. Run the speech normalizer and post-process the output.
```bash
mkdir -p ${RESULTS_PATH}

python examples/speech_recognition/new/infer.py \
    --config-dir examples/hubert/config/decode/ \
    --config-name infer_viterbi \
    task.data=${DATA_DIR} \
    task.normalize=false \
    common_eval.results_path=${RESULTS_PATH}/log \
    common_eval.path=${DATA_DIR}/checkpoint_best.pt \
    dataset.gen_subset=${GEN_SUBSET} \
    '+task.labels=["unit"]' \
    +decoding.results_path=${RESULTS_PATH} \
    common_eval.post_process=none \
    +dataset.batch_size=1 \
    common_eval.quiet=True

# Post-process and generate output at ${RESULTS_PATH}/${GEN_SUBSET}.txt
python examples/speech_to_speech/preprocessing/prep_sn_output_data.py \
  --in-unit ${RESULTS_PATH}/hypo.units \
  --in-audio ${DATA_DIR}/${GEN_SUBSET}.tsv \
  --output-root ${RESULTS_PATH}
```


### Unit-to-waveform conversion with unit vocoder
The pre-trained vocoders can support generating audio for both full unit sequences and reduced unit sequences (i.e. duplicating consecutive units removed). Set `--dur-prediction` for generating audio with reduced unit sequences.
```bash
# IN_CODE_FILE contains one unit sequence per line. Units are separated by space.

python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${IN_CODE_FILE} \
  --vocoder ${VOCODER_CKPT} --vocoder-cfg ${VOCODER_CFG} \
  --results-path ${RESULTS_PATH} --dur-prediction
```

## Training new models
To be updated.
