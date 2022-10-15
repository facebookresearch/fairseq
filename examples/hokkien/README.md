# English-Hokkien Speech-to-speech Translation

We open source the speech-to-speech translation (S2ST) system to translate between English and Taiwanese Hokkien (hereafter Hokkien). Including the models we build, and the English-Hokkien parallel speech-to-speech translation benchmark dataset (TAT-S2ST) that we created to facilitate future research in this field.

Details about building the model and dataset can be found in "[Speech-to-speech translation for a real-world unwritten language](https://research.facebook.com/publications/hokkien-direct-speech-to-speech-translation)"

## Dataset: TAT Speech-to-speech translation benchmark dataset (TAT-S2ST)

We create and release a Hokkien-English parallel speech dataset that is available for benchmarking Hokkien<>English speech to speech translation systems. The dataset was derived from TAT-Vol1-eval-lavalier (dev) and TAT-Vol1-test-lavalier (test) based on [Taiwanese Across Taiwan (TAT) corpus](https://sites.google.com/speech.ntut.edu.tw/fsw/home/tat-corpus), which contained audio recordings and transcripts in Taiwanese Hokkien.
We created the parallel dataset by first concatenating neighboring sentences to form longer utterances, translating the Hokkien text transcriptions into English via Hokkien-English bilinguals, and recording the English translations with human voices. Below are some summary statistics of the dataset.

The dataset is available [HERE](https://sites.google.com/nycu.edu.tw/speechlabx/tat_s2st_benchmark).

## Open Sourced English-Hokkien S2ST Models

We open source our best S2ST models, which include two components, the speech-to-unit model to translate the source speech to discrete unit in target language, and the unit vocoder to further convert into target speech in waveform.

### Speech-to-unit models

English->Hokkien

| Model Type | Model ID | Link | ASR-BLEU <br> TAT-S2ST Test |
|-|-|-|-|
| Single-pass decoder (S2UT) | xm_transformer_s2ut_en-hk | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/xm_transformer_s2ut_en-hk.tar.gz) | 5.99 |
| Two-pass decoder (UnitY) | xm_transformer_unity_en-hk | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/xm_transformer_unity_en-hk.tar.gz) | 7.47 |

Hokkien->English

| Model Type | Model ID | Link | ASR-BLEU <br> TAT-S2ST Test |
|-|-|-|-|
| Single-pass decoder (S2UT) | xm_transformer_s2ut_hk-en | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/xm_transformer_s2ut_hk-en.tar.gz) | 8.10 |
| Two-pass decoder (UnitY) | xm_transformer_unity_hk-en | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/xm_transformer_unity_hk-en.tar.gz) | 14.12 |

### Unit-based HiFi-GAN vocoder

| Language | Checkpoint |
|-|-|
| English * | [Download](http://dl.fbaipublicfiles.com/fairseq/vocoder/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur.tar.gz) |
| Hokkien | [Download](http://dl.fbaipublicfiles.com/fairseq/vocoder/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS.tar.gz) |

(*) The English unit vocoder was open sourced from our previous work, [Textless S2ST with Real Data](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/textless_s2st_real_data.md#unit-based-hifi-gan-vocoder).

## Inference
```
python3 torchhub_tts_prediction.py \
    --model-id <model_id_from_above> \
    --input-audio <input_audio_path> \
    --output-audio <output_audio_path>
```

## ASR-BLEU Evaluation

See [ASR-BLEU example page](https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_to_speech/asr_bleu) about the script to evaluate S2ST models.


## Citation
```
@inproceedings{enhokkien,
    title = "Speech-to-speech translation for a real-world unwritten language",
    author = "Peng-Jen Chen and
      Kevin Tran and
      Yilin Yang and
      Jingfei Du and
      Justine Kao and
      Yu-An Chung and
      Paden Tomasello and
      Paul-Ambroise Duquenne and
      Holger Schwenk and
      Hongyu Gong and
      Hirofumi Inaguma and
      Sravya Popuri and
      Changhan Wang and
      Juan Pino and 
      Wei-Ning Hsu and 
      Ann Lee",
}
```

## License

The released models and TAT-S2ST dataset are under NC-BY-CC 4.0.
