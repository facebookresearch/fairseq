# Speech Recognition
`examples/speech_recognition` is implementing ASR task in Fairseq, along with needed features, datasets, models and loss functions to train and infer model described in [Transformers with convolutional context for ASR (Abdelrahman Mohamed et al., 2019)](https://arxiv.org/abs/1904.11660).


## Additional dependencies
On top of main fairseq dependencies there are couple more additional requirements.

1) Please follow the instructions to install [torchaudio](https://github.com/pytorch/audio). This is required to compute audio fbank features.
2) [Sclite](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm#sclite_name_0) is used to measure WER. Sclite can be downloaded and installed from source from sctk package [here](http://www.openslr.org/4/). Training and inference doesn't require Sclite dependency.

## Preparing librispeech data
```
./examples/speech_recognition/datasets/prepare-librispeech.sh $DIR_TO_SAVE_RAW_DATA $DIR_FOR_PREPROCESSED_DATA
```

## Training librispeech data
```
python train.py $DIR_FOR_PREPROCESSED_DATA --save-dir $MODEL_PATH --max-epoch 80 --task speech_recognition --arch vggtransformer_2 --optimizer adadelta --lr 1.0 --adadelta-eps 1e-8 --adadelta-rho 0.95 --clip-norm 10.0  --max-tokens 5000 --log-format json --log-interval 1 --criterion cross_entropy_acc --user-dir examples/speech_recognition/
```

## Inference for librispeech
`$SET` can be `test_clean` or `test_other`
Any checkpoint in `$MODEL_PATH` can be selected. In this example we are working with `checkpoint_last.pt`
```
python examples/speech_recognition/infer.py $DIR_FOR_PREPROCESSED_DATA --task speech_recognition --max-tokens 25000 --nbest 1 --path $MODEL_PATH/checkpoint_last.pt --beam 20 --results-path $RES_DIR --batch-size 40 --gen-subset $SET --user-dir examples/speech_recognition/
```

## Inference for librispeech
```
sclite -r ${RES_DIR}/ref.word-checkpoint_last.pt-${SET}.txt -h ${RES_DIR}/hypo.word-checkpoint_last.pt-${SET}.txt -i rm -o all stdout > $RES_REPORT
```
`Sum/Avg` row from first table of the report has WER
