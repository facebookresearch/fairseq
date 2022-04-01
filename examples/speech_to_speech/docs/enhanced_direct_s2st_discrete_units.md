# Speech to speech translation (S2ST)

We provide the implementation for speech-to-unit translation (S2UT) proposed in [_Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation_](arxiv link) and the various pretrained models used.

## Pretrained Models

### Unit extraction 
We used the multilingual HuBERT model open sourced in [Textless S2ST with Real Data](textless_s2st_real_data.md)

### Wav2vec 2.0
Language | Block type | Model size | Dataset | Model |
--- | --- | --- | --- | --- |
Es | Transformer | BASE | Voxpopuli | [ckpt ](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/es/transformer_B.pt) |
Es | Transformer | LARGE | Voxpopuli | [ckpt ](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/es/transformer_L.pt) |
Es | Conformer | LARGE | Voxpopuli | [ckpt ](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/es/conformer_L.pt) |
En | Transformer | BASE | Librilight| [ckpt ](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/en/transformer_B.pt) |
En | Conformer | LARGE | Librilight | [ckpt ](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/w2v2/en/conformer_L.pt) |

### Unit mBART
Unit size | Dataset | Unit config | Model |
--- | --- | --- | --- | 
1000 | [Voxpopuli](https://aclanthology.org/2021.acl-long.80) En, Es unlabelled speech  | [mbart_large](https://github.com/pytorch/fairseq/blob/f591cc94caa85098ccf125a4782f91125b6a086d/fairseq/models/bart/model.py#L368) |[ckpt](htpps://dl.fbaipublicfiles.com/fairseq/speech_to_speech/s2st_finetuning/unit_mBART/checkpoint.pt) |


## Data preparation 

1. To prepare data for S2UT finetuning, follow the steps from [Direct S2ST with Discrete Units](./direct_s2st_discrete_units.md) and format the data in the _S2UT_ format. Note that we use 1000 units from the eleventh layer (`--layer 11`) of the multilingual hubert model linked above instead
2. Run
```
var="id\taudio\tn_frames\ttgt_text\ttgt_n_frames"
sed -i "1s/.*/$var/" ${SPLIT}.tsv
```

## Training

**Speech-to-unit translation (S2UT)**

Here's an example for finetuning S2UT models with 1000 discrete units as target. You can download the config file and vocabulary from here[add links]:
```
fairseq-train $DATA_ROOT \
  --config-yaml config.yaml  \
  --task speech_to_text --arch xm_transformer\
  --criterion l --label-smoothing 0.2 \
  --share-decoder-input-output-embed --adaptor-n-layers 1 --normalize\
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --load-pretrained-decoder-from ${unit_mBART} --w2v-path ${wav2vec2.0} \
  --mask-prob 0.3 --mask-channel-length 32 --mask-channel-prob 0.25\
  --save-dir ${MODEL_DIR} --checkpoint-activations --encoder-proj \
  --lr 0.0005 --dropout 0.1 --attention-dropout 0.1 --lr-scheduler inverse_sqrt\
  --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
  --max-update 20000 --max-tokens 4000 --max-tokens-valid 4000 --max-source-positions 4000 \
  --max-target-positions 4000 --update-freq 120 \
  --seed 1 --fp16 --num-workers 1
```
* Adjust `--update-freq` accordingly for different #GPUs. In the above we set `--update-freq 15` to simulate training with 120 GPUs.
* In the above setting we finetune the model end to end, corresponding to the full setup in the paper.
* To apply LNA-E partial finetuning, add `--finetune-w2v-params layer_norm,self_attn`
* For LNA-D partial finetuning add `--finetune-decoder-params encoder_attn,layer_norm,self_attn`. To optionally freeze the encoder by k updates, use `--freeze-finetune-updates ${K}`
* For LNA-E,D partial finetuning add both the above options.


**Unit-based HiFi-GAN vocoder** 

We apply the open-sourced unit-based HiFi-GAN vocoders to convert the predicted unit sequences to waveform. They are open sourced in [Textless S2ST with Real Data](textless_s2st_real_data.md)

## Inference

**Speech-to-unit translation (S2UT)**

1. Follow the same inference process as in [fairseq-S2T](https://github.com/pytorch/fairseq/tree/main/examples/speech_to_text) to generate unit sequences (`${RESULTS_PATH}/generate-${GEN_SUBSET}.txt`).
```
fairseq-generate $DATA_ROOT \
  --config-yaml config.yaml \
  --task speech_to_text  \
  --path $MODEL_DIR/checkpoint_best.pt  --gen-subset $GEN_SUBSET \
  --max-tokens 10000 --max-source-positions 10000 --max-target-positions 10000\
  --beam 10 --max-len-a 1 --max-len-b 200 \
  --results-path ${RESULTS_PATH}
```

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


## Evaluation

To evaluate speech translation output, we first apply ASR on the speech output and then compute BLEU score betweent the ASR decoded text and the references using sacreBLEU.

* Text normalization: We use the text cleaner at [https://github.com/keithito/tacotron](https://github.com/keithito/tacotron) for pre-processing reference English text for ASR BLEU evaluation. The text cleaner used for Spanish text normalization will be updated here shortly.
* En ASR: We use the "[Wav2Vec 2.0 Large (LV-60) + Self Training / 960 hours / Libri-Light + Librispeech](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt)" En ASR model open-sourced by the [wav2vec](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) project. The model is also available on [Hugging Face](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).
* Es ASR: We use the [Wav2Vec2-Large-XLSR-53-Spanish](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) finetuned on spanish Common Voice Es ASR model open-sourced by Jonatasgrosman(https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish) on [Hugging Face](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish).
* See [instructions](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#evaluating-a-ctc-model) on how to run inference with a wav2vec-based ASR model.
