# End-to-end NLU

End-to-end spoken language understanding (SLU) predicts intent directly from audio using a single model. It promises to improve the performance of assistant systems by leveraging acoustic information lost in the intermediate textual representation and preventing cascading errors from Automatic Speech Recognition (ASR). Further, having one unified model has efficiency advantages when deploying assistant systems on-device.

This page releases the code for reproducing the results in [STOP: A dataset for Spoken Task Oriented Semantic Parsing](https://arxiv.org/abs/2207.10643)

The dataset can be downloaded here: [download link](https://dl.fbaipublicfiles.com/stop/stop.tar.gz)

The low-resource splits can be downloaded here: [download link](http://dl.fbaipublicfiles.com/stop/low_resource_splits.tar.gz)

## Pretrained models end-to-end NLU Models

| Speech Pretraining | ASR Pretraining | Test EM Accuracy | Tesst EM-Tree Accuracy | Link |
| ----------- | ----------- |----------|----------|----------|
| None   | None | 36.54 | 57.01 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-none-none.pt) |
| Wav2Vec   | None | 68.05 | 82.53 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-wav2vec-none.pt) |
| HuBERT   | None | 68.40 | 82.85 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-hubert-none.pt) |
| Wav2Vec   | STOP | 68.70 | 82.78 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-wav2vec-stop.pt) |
| HuBERT   | STOP | 69.23 | 82.87 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-hubert-stop.pt) |
| Wav2Vec   | Librispeech | 68.47 | 82.49 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-wav2vec-ls.pt) |
| HuBERT   | Librispeech | 68.70 | 82.78 | [link](https://dl.fbaipublicfiles.com/stop/end-to-end-nlu-hubert-ls.pt) |

## Pretrained models ASR Models
| Speech Pre-training  | ASR Dataset | STOP Eval WER | STOP Test WER | dev\_other WER | dev\_clean WER | test\_clean WER | test\_other WER | Link |
| ----------- |  ----------- |  ----------- |  ----------- |  ----------- |  ----------- |  ----------- |  ----------- |  ----------- |
| HuBERT  | Librispeech | 8.47 | 2.99 | 3.25 | 8.06 | 25.68 | 26.19 | [link](https://dl.fbaipublicfiles.com/stop/ctc-asr-hubert-ls.pt) |
| Wav2Vec  | Librispeech | 9.215 | 3.204 | 3.334 | 9.006 | 27.257 | 27.588 | [link](https://dl.fbaipublicfiles.com/stop/ctc-asr-wav2vec-ls.pt) |
| HuBERT  | STOP | 46.31 | 31.30 | 31.52 | 47.16 | 4.29 | 4.26 | [link](https://dl.fbaipublicfiles.com/stop/ctc-asr-hubert-stop.pt) |
| Wav2Vec  | STOP | 43.103 | 27.833 | 28.479 | 28.479 | 4.679 | 4.667 | [link](https://dl.fbaipublicfiles.com/stop/ctc-asr-wav2vec-stop.pt) |
| HuBERT  | Librispeech + STOP | 9.015 | 3.211 | 3.372 | 8.635 | 5.133 | 5.056 | [link](https://dl.fbaipublicfiles.com/stop/ctc-asr-hubert-ls-stop.pt) |
| Wav2Vec  | Librispeech + STOP | 9.549 | 3.537 | 3.625 | 9.514 | 5.59 | 5.562 | [link](https://dl.fbaipublicfiles.com/stop/ctc-asr-wav2vec-ls-stop.pt) |

## Creating the fairseq datasets from STOP

First, create the audio file manifests and label files:

```
python examples/audio_nlp/nlu/generate_manifests.py --stop_root $STOP_DOWNLOAD_DIR/stop --output $FAIRSEQ_DATASET_OUTPUT/
```


Run `./examples/audio_nlp/nlu/create_dict_stop.sh $FAIRSEQ_DATASET_OUTPUT` to generate the fairseq dictionaries.


## Training an End-to-end NLU Model


Download a wav2vec or hubert model from [link](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) or [link](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)


```
python fairseq_cli/hydra-train  --config-dir examples/audio_nlp/nlu/configs/  --config-name nlu_finetuning task.data=$FAIRSEQ_DATA_OUTPUT model.w2v_path=$PRETRAINED_MODEL_PATH
```
