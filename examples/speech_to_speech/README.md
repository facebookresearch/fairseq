# Speech to speech translation (S2ST)

We provide the implementation for speech-to-unit translation (S2UT) proposed in "[Direct speech-to-speech translation with discrete units (Lee et al. 2021)](https://arxiv.org/abs/2107.05604)" and also the transformer-based implementation of the speech-to-spectrogram translation (S2SPECT, or transformer-based [Translatotron](https://arxiv.org/abs/1904.06037)) baseline in the paper.

## Pretrained Models

### Unit-based HiFi-GAN Vocoder
Unit config | Unit size | Vocoder dataset | Model
|---|---|---|---
[HuBERT Base, Librispeech](https://github.com/fairinternal/fairseq-py/tree/main/examples/hubert), layer 6 | 100 | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json)


## Data preparation
### Target speech
0. (optional) To prepare S2S data from a speech-to-text translation (ST) dataset, see [fairseq-S^2](https://github.com/pytorch/fairseq/tree/main/examples/speech_synthesis) for pre-trained TTS models and instructions on how to train and decode TTS models.
1. Prepare two folders, `$SRC_AUDIO` and `$TGT_AUDIO`, with `${SPLIT}/${SAMPLE_ID}.wav` for source and target speech under each folder, separately. Note that for S2UT experiments, target audio sampling rate should be in 16,000 Hz, and for S2SPECT experiments, target audio sampling rate is recommended to be in 22,050 Hz.
2. To prepare target discrete units for S2UT model training, see [Generative Spoken Language Modeling (speech2unit)](https://github.com/pytorch/fairseq/tree/main/examples/textless_nlp/gslm/speech2unit) for pre-trained k-means models, checkpoints, and instructions on how to decode units from speech. Set the output target unit files (`--out_quantized_file_path`) as `${TGT_AUDIO}/${SPLIT}.txt`. In [Lee et al. 2021](https://arxiv.org/abs/2107.05604), we use 100 units from the sixth layer (`--layer 6`) of the HuBERT Base model.

### Formatting data
**Speech-to-speech data**

_S2UT_
  * Set `--reduce-unit` for training S2UT _reduced_ model
  * Pre-trained vocoder and config (`$VOCODER_CKPT`, `$VOCODER_CFG`) can be downloaded from the **Pretrained Models** section. They are not required if `--eval-inference` is not going to be set during model training.
```
# $SPLIT1, $SPLIT2, etc. are split names such as train, dev, test, etc.

python examples/speech_to_speech/preprocessing/prep_s2ut_data.py \
  --source-dir $SRC_AUDIO --target-dir $TGT_AUDIO --data-split $SPLIT1 $SPLIT2 \
  --output-root $DATA_ROOT --reduce-unit \
  --vocoder-checkpoint $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG
```

_S2SPECT_
```
# $SPLIT1, $SPLIT2, etc. are split names such as train, dev, test, etc.

python examples/speech_to_speech/preprocessing/prep_s2spect_data.py \
  --source-dir $SRC_AUDIO --target-dir $TGT_AUDIO --data-split $SPLIT1 $SPLIT2 \
  --output-root $DATA_ROOT
```

**Multitask data**
  * For each multitask `$TASK_NAME`, prepare `${DATA_ROOT}/${TASK_NAME}/${SPLIT}.tsv` files for each split following the format below: (Two tab separated columns. The sample_ids should match with the sample_ids for the speech-to-speech data in `${DATA_ROOT}/${SPLIT}.tsv`.)
```
id  tgt_text
sample_id_0 token1 token2 token3 ...
sample_id_1 token1 token2 token3 ...
...
```
  * For each multitask `$TASK_NAME`, prepare `${DATA_ROOT}/${TASK_NAME}/dict.txt`, a dictionary in fairseq format with all tokens for the targets for `$TASK_NAME`.
  * Create `config_multitask.yaml`. Below is an example of the config used for S2UT _reduced_ with Fisher experiments including two encoder multitasks (`source_letter`, `target_letter`) and one decoder CTC task (`decoder_target_ctc`).
```
source_letter:  # $TASK_NAME
   decoder_type: transformer
   dict: ${DATA_ROOT}/source_letter/dict.txt
   data: ${DATA_ROOT}/source_letter
   encoder_layer: 6
   loss_weight: 8.0
target_letter:
   decoder_type: transformer
   dict: ${DATA_ROOT}/target_letter/dict.txt
   data: ${DATA_ROOT}/target_letter
   encoder_layer: 8
   loss_weight: 8.0
decoder_target_ctc:
   decoder_type: ctc
   dict: ${DATA_ROOT}/decoder_target_ctc/dict.txt
   data: ${DATA_ROOT}/decoder_target_ctc
   decoder_layer: 3
   loss_weight: 1.6
```


## Training

**Speech-to-unit translation (S2UT)**

Here's an example for training Fisher S2UT models with 100 discrete units as target:
```
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan  \
  --criterion speech_to_unit --label-smoothing 0.2 \
  --arch s2ut_transformer_fisher --share-decoder-input-output-embed \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${MODEL_DIR} \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 400000 --max-tokens 20000 --max-target-positions 3000 --update-freq 4 \
  --seed 1 --fp16 --num-workers 8
```
* Adjust `--update-freq` accordingly for different #GPUs. In the above we set `--update-freq 4` to simulate training with 4 GPUs.
* Set `--n-frames-per-step 5` to train an S2UT _stacked_ system with reduction ratio r=5. (Use `$DATA_ROOT` prepared without `--reduce-unit`.)
* (optional) one can turn on tracking MCD loss during training for checkpoint selection by setting `--eval-inference --eval-args '{"beam": 1, "max_len_a": 1}' --best-checkpoint-metric mcd_loss`. It is recommended to sample a smaller subset as the validation set as MCD loss computation is time-consuming.

**Speech-to-spectrogram translation (S2SPECT)**

Here's an example for training Fisher S2SPECT models with reduction ratio r=5:
```
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --n-frames-per-step 5 \
  --criterion speech_to_spectrogram \
  --arch s2spect_transformer_fisher --decoder-normalize-before \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --save-dir ${MODEL_DIR} \
  --eval-inference --best-checkpoint-metric mcd_loss \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 --weight-decay 1e-6 \
  --max-update 400000 --max-tokens 80000 --max-tokens-valid 30000  --required-batch-size-multiple 1 \
  --max-target-positions 3000 --update-freq 16 \
  --seed 1 --fp16 --num-workers 8
```
* Adjust `--update-freq` accordingly for different #GPUs. In the above we set `--update-freq 16` to simulate training with 16 GPUs.
* We recommend turning on MCD loss during training for the best checkpoint selection.

**Unit-based HiFi-GAN vocoder**

The vocoder is trained with the [speech-resynthesis repo](https://github.com/facebookresearch/speech-resynthesis). See [here](https://github.com/facebookresearch/speech-resynthesis/tree/main/examples/speech_to_speech_translation) for instructions on how to train the unit-based HiFi-GAN vocoder with duration prediction. The same vocoder can support waveform generation for both _reduced_ unit sequences (with `--dur-prediction` set during inference) and original unit sequences.

## Inference

**Speech-to-unit translation (S2UT)**

1. Follow the same inference process as in [fairseq-S2T](https://github.com/pytorch/fairseq/tree/main/examples/speech_to_text) to generate unit sequences (`${RESULTS_PATH}/generate-${GEN_SUBSET}.txt`).
```
fairseq-generate $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan \
  --path $MODEL_DIR/checkpoint_best.pt  --gen-subset $GEN_SUBSET \
  --max-tokens 50000 \
  --beam 10 --max-len-a 1 \
  --results-path ${RESULTS_PATH}
```
  * Set `--beam 1 --n-frames-per-step $r` for decoding with S2UT _stacked_ models.

2. Convert unit sequences to waveform.
```
grep "^D\-" ${RESULTS_PATH}/generate-${GEN_SUBSET}.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit

python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${RESULTS_PATH} --dur-prediction
```
 * Set `--dur-prediction` for generating audio for S2UT _reduced_ models.


**Speech-to-spectrogram translation (S2SPECT)**

Follow the same inference process as in [fairseq-S^2](https://github.com/pytorch/fairseq/tree/main/examples/speech_synthesis) to generate waveform.

```
# assume using a default Griffin-Lim vocoder

python examples/speech_synthesis/generate_waveform.py $DATA_ROOT \
  --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
  --task speech_to_speech --n-frames-per-step 5 \
  --path $MODEL_DIR/checkpoint_best.pt  --gen-subset $GEN_SUBSET \
  --max-tokens 50000 \
  --results-path ${RESULTS_PATH} --dump-waveforms --output-sample-rate 16000
```

In addition to using the default Griffin-Lim vocoder, one can also finetune a HiFi-GAN vocoder for the S2SPECT model by following the instructions in the [HiFi-GAN repo](https://github.com/jik876/hifi-gan).

**Multitask decoding**

Coming soon.

## Evaluation

To evaluate speech translation output, we first apply ASR on the speech output and then compute BLEU score betweent the ASR decoded text and the references using sacreBLEU.

**En**
* ASR: We use the "[Wav2Vec 2.0 Large (LV-60) + Self Training / 960 hours / Libri-Light + Librispeech](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt)" En ASR model open-sourced by the [wav2vec](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) project. See [instructions](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#evaluating-a-ctc-model) on how to run inference with a wav2vec-based ASR model. The model is also available on [Hugging Face](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).
* Text normalization: We use the text cleaner at [https://github.com/keithito/tacotron](https://github.com/keithito/tacotron) for pre-processing reference English text for ASR BLEU evaluation.
