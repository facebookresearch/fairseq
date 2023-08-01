### 2021 Update: We are merging this example into the [S2T framework](../speech_to_text), which supports more generic speech-to-text tasks (e.g. speech translation) and more flexible data processing pipelines. Please stay tuned.

# Speech Recognition
`examples/speech_recognition` is implementing ASR task in Fairseq, along with needed features, datasets, models and loss functions to train and infer model described in [Transformers with convolutional context for ASR (Abdelrahman Mohamed et al., 2019)](https://arxiv.org/abs/1904.11660).


## Additional dependencies
On top of main fairseq dependencies there are couple more additional requirements.

1) Please follow the instructions to install [torchaudio](https://github.com/pytorch/audio). This is required to compute audio fbank features.
2) [Sclite](http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm#sclite_name_0) is used to measure WER. Sclite can be downloaded and installed from source from sctk package [here](http://www.openslr.org/4/). Training and inference doesn't require Sclite dependency.
3) [sentencepiece](https://github.com/google/sentencepiece) is required in order to create dataset with word-piece targets.

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

## Using flashlight (previously called [wav2letter](https://github.com/facebookresearch/wav2letter)) components
[flashlight](https://github.com/facebookresearch/flashlight) now has integration with fairseq. Currently this includes:

* AutoSegmentationCriterion (ASG)
* flashlight-style Conv/GLU model
* flashlight's beam search decoder

To use these, follow the instructions on [this page](https://github.com/flashlight/flashlight/tree/e16682fa32df30cbf675c8fe010f929c61e3b833/bindings/python) to install python bindings. **Flashlight v0.3.2** must be used to install the bindings. Running:
```
git clone --branch v0.3.2 https://github.com/flashlight/flashlight
```
will properly clone and check out this version.

## Training librispeech data (flashlight style, Conv/GLU + ASG loss)
Training command:
```
python train.py $DIR_FOR_PREPROCESSED_DATA --save-dir $MODEL_PATH --max-epoch 100 --task speech_recognition --arch w2l_conv_glu_enc --batch-size 4 --optimizer sgd --lr 0.3,0.8 --momentum 0.8 --clip-norm 0.2 --max-tokens 50000 --log-format json --log-interval 100 --num-workers 0 --sentence-avg --criterion asg_loss --asg-transitions-init 5 --max-replabel 2 --linseg-updates 8789 --user-dir examples/speech_recognition
```

Note that ASG loss currently doesn't do well with word-pieces. You should prepare a dataset with character targets by setting `nbpe=31` in `prepare-librispeech.sh`.

## Inference for librispeech (flashlight decoder, n-gram LM)
Inference command:
```
python examples/speech_recognition/infer.py $DIR_FOR_PREPROCESSED_DATA --task speech_recognition --seed 1 --nbest 1 --path $MODEL_PATH/checkpoint_last.pt --gen-subset $SET --results-path $RES_DIR --w2l-decoder kenlm --kenlm-model $KENLM_MODEL_PATH --lexicon $LEXICON_PATH --beam 200 --beam-threshold 15 --lm-weight 1.5 --word-score 1.5 --sil-weight -0.3 --criterion asg_loss --max-replabel 2 --user-dir examples/speech_recognition
```

`$KENLM_MODEL_PATH` should be a standard n-gram language model file. `$LEXICON_PATH` should be a flashlight-style lexicon (list of known words and their spellings). For ASG inference, a lexicon line should look like this (note the repetition labels):
```
doorbell  D O 1 R B E L 1 ▁
```
For CTC inference with word-pieces, repetition labels are not used and the lexicon should have most common spellings for each word (one can use sentencepiece's `NBestEncodeAsPieces` for this):
```
doorbell  ▁DOOR BE LL
doorbell  ▁DOOR B E LL
doorbell  ▁DO OR BE LL
doorbell  ▁DOOR B EL L
doorbell  ▁DOOR BE L L
doorbell  ▁DO OR B E LL
doorbell  ▁DOOR B E L L
doorbell  ▁DO OR B EL L
doorbell  ▁DO O R BE LL
doorbell  ▁DO OR BE L L
```
Lowercase vs. uppercase matters: the *word* should match the case of the n-gram language model (i.e. `$KENLM_MODEL_PATH`), while the *spelling* should match the case of the token dictionary (i.e. `$DIR_FOR_PREPROCESSED_DATA/dict.txt`).

## Inference for librispeech (flashlight decoder, viterbi only)
Inference command:
```
python examples/speech_recognition/infer.py $DIR_FOR_PREPROCESSED_DATA --task speech_recognition --seed 1 --nbest 1 --path $MODEL_PATH/checkpoint_last.pt --gen-subset $SET --results-path $RES_DIR --w2l-decoder viterbi --criterion asg_loss --max-replabel 2 --user-dir examples/speech_recognition
```
