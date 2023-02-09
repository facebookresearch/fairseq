# Generative Spoken Dialogue Language Modeling
[[paper]](https://arxiv.org/abs/2203.16502) [[demo samples]](https://speechbot.github.io/dgslm/index.html) [[blog]](https://ai.facebook.com/blog/generating-chit-chat-including-laughs-yawns-ums-and-other-nonverbal-cues-from-raw-audio/)

This repo contains the code and pre-trained models for the paper _Generative Spoken Dialogue Language Modeling_.
<details>
  <summary>Paper abstract </summary>

> We introduce dGSLM, the first "textless" model able to generate audio samples of naturalistic spoken dialogues. It uses recent work on unsupervised spoken unit discovery coupled with a dual-tower transformer architecture with cross-attention trained on 2000 hours of two-channel raw conversational audio (Fisher dataset) without any text or labels. We show that our model is able to generate speech, laughter and other paralinguistic signals in the two channels simultaneously and reproduces more naturalistic and fluid turn taking compared to a text-based cascaded model.

</details>

## [Speech-to-Unit Encoder for dGSLM: The Fisher HuBERT model](hubert_fisher/)
The [hubert_fisher](hubert_fisher/) repository contains the pre-trained models and recipies to produce discrete units for the dGSLM model.

## [Unit-to-Speech Decoder for dGSLM](vocoder_hifigan/)
The [vocoder_hifigan](vocoder_hifigan/) repo contains the vocoder and recipies to synthesize the waveform from the discrete units.

## Spoken Dialogue Transformer Language Model (SpeechDLM)
### Pre-trained model
We share the pre-trained model checkpoint for the best configuration in the paper (DLM-5 model, with Edge Unit Prediction & Delayed Duration Prediction objectives), dubbed as `SpeechDLM`, trained on the 2000 hours of Fisher dataset :
| Pre-trained SpeechDLM model trained on Fisher dataset |
|-----------------------------------------------|
|[model checkpoint](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/speech_dlm/speech_dlm_base.pt) - [dictionary 1](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/speech_dlm/dict.unitA.txt) - [dictionary 2](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/speech_dlm/dict.unitB.txt)|
the two dictionary files correspond to the two channels, and actually have the same content.

### Sample from a trained model
You can sample from a trained SpeechDLM model interactively :
```python
from fairseq.models.speech_dlm import SpeechDLM

# Load SpeechDLM model
speech_dlm = SpeechDLM.from_pretrained(
                model_name_or_path='/path/to/model/dir',
                checkpoint_file='speech_dlm_base.pt',
                data_name_or_path='/path/to/data/dir'
            )
# Disable dropout
speech_dlm.eval()
# Move model to GPU
speech_dlm.cuda()

# Define the input sequences
input_sequences = [{
      'unitA': '7 376 376 133 178 486 486 486 486 486 486 486 486 2 486',
      'unitB': '7 499 415 177 7 7 7 7 7 7 136 136 289 289 408'
    }]

# Sample from the SpeechDLM model
generated_units = speech_dlm.sample(
        input_sequences,
        max_len_a = 0,
        max_len_b = 500,
        sampling=True,
        beam=5,
    )
# >> {'unitA': '7 376 376 133 178 486 486 486 486 486 486 486 486 2 486 486 178 486 486 2 2 376 376 486 486 486 376 376 387 387 ...',
# >> 'unitB': '7 499 415 177 7 7 7 7 7 7 136 136 289 289 408 32 428 95 356 141 331 439 350 350 192 331 445 202 104 104 ...'}
```

Or using the `sample_speech_dlm.py` script :
```bash
python sample_speech_dlm.py \
    --in-file $INPUT_CODE_FILE --out-file $OUTPUT_FILE \
    --ckpt $CHECKPOINT_PATH --data $DATA_DIR
```
where each line of INPUT_CODE_FILE is a dictionary with keys `'audio', 'unitA', 'unitB'` as follows :
```
{'audio': 'file_1', 'unitA': '8 8 ... 352 352', 'unitB': '217 8 ... 8 8'}
{'audio': 'file_2', 'unitA': '5 5 ... 65 65', 'unitB': '6 35 ... 8 9'}
...
```
This code file can be created with the script `create_input_code.py` (using the outputs of `quantize_with_kmeans.py` [here](hubert_fisher/#encode-audio-to-discrete-units)) :
```bash
python examples/textless_nlp/dgslm/vocoder_hifigan/create_input_code.py \
    $CHANNEL1_UNITS $CHANNEL2_UNITS $OUTPUT_CODE_FILE
```

### Training a SpeechDLM model
#### 1) Data preparation
First, you need to prepare the raw dataset. For each `split` (train, valid), you need two files corresponding to two channels (namely `unitA` and `unitB` for example) containing the units from each channel separately. Make sure that 2 files have the same number of lines and each corresponding line has the same number of units.

Here is an example of `.unitA` file :
```
7 376 376 133 178
486 486 486
486 376
```
and the corresponding `.unitB` file :
```
7 499 415 177 7
7 7 136
331 445
```
These two files can be obtained using the [example command](hubert_fisher/#encode-audio-to-discrete-units) of hubert fisher, with the `--hide-fname` option added.

The raw dataset directory should contain the following files :
```
train.unitA valid.unitA
train.unitB valid.unitB
```

Next preprocess/binarize the data with `fairseq-preprocess`, but make sure to preprocess each channel separately, and **rename** the preprocessed files under the following format `${split}.${channel}.{bin, idx}`. Each channel also needs a separate dictionary file under the name `dict.${channel}.txt` .

Here is an example pre-processing code :

```bash
# Preprocess the first channel (unitA)
fairseq-preprocess --source-lang unitA \
    --only-source \
    --trainpref $RAW_DATA_DIR/train \
    --validpref $RAW_DATA_DIR/valid \
    --destdir $BIN_DATA_DIR \
    --workers 20

# Preprocess the second channel (unitB) and reuse the dictionary from the first channel
fairseq-preprocess --source-lang unitB \
    --srcdict $BIN_DATA_DIR/dict.unitA.txt \
    --only-source \
    --trainpref $RAW_DATA_DIR/train \
    --validpref $RAW_DATA_DIR/valid \
    --destdir $BIN_DATA_DIR \
    --workers 20

# Rename the bin & index files
for channel in unitA unitB; do
  for split in train valid; do
    mv $BIN_DATA_DIR/${split}.${channel}-None.${channel}.bin $BIN_DATA_DIR/${split}.${channel}.bin
    mv $BIN_DATA_DIR/${split}.${channel}-None.${channel}.idx $BIN_DATA_DIR/${split}.${channel}.idx
  done
done
```
Finally, the preprocessed (bin) dataset directory should contain the following files :
```
dict.unitA.txt  train.unitA.idx train.unitA.bin valid.unitA.idx valid.unitA.bin
dict.unitB.txt  train.unitB.idx train.unitB.bin valid.unitB.idx valid.unitB.bin
```

#### 2) Train the model
To train the SpeechDLM (with the configuration as the pre-trained model) on 2 GPUs :
```bash
fairseq-train $BIN_DATA_DIR \
    --save-dir $CHECKPOINT_DIR \
    --tensorboard-logdir $CHECKPOINT_DIR \
    --task speech_dlm_task --channels unitA,unitB \
    --next-unit-prediction "False" --edge-unit-prediction "True" \
    --duration-prediction "True" --delayed-duration-target "True" \
    --criterion speech_dlm_criterion \
    --arch speech_dlm --decoder-cross-layers 4 \
    --share-decoder-input-output-embed \
    --dropout 0.1 --attention-dropout 0.1 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
    --max-tokens 18432 --tokens-per-sample 6144 --sample-break-mode none \
    --update-freq 16 --num-workers 4 --skip-invalid-size-inputs-valid-test \
    --max-update 250000 --warmup-updates 20000 \
    --save-interval-updates 10000 --keep-last-epochs 1 --no-epoch-checkpoints \
    --log-interval 50 --seed 100501 \
    --fp16 --checkpoint-activations
```

#### 3) Validate
The model can be validated via the `fairseq-validate` command :
```bash
fairseq-validate $BIN_DATA_DIR \
    --task speech_dlm_task \
    --path $CHECKPOINT_PATH \
    --max-tokens 6144
```

## Reference

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{nguyen2022dgslm,
  title   = {Generative Spoken Dialogue Language Modeling},
  author  = {Nguyen, Tu Anh and Kharitonov, Eugene and Copet, Jade and Adi, Yossi and Hsu, Wei-Ning and Elkahky, Ali and Tomasello, Paden and Algayres, Robin and Sagot, Benoit and Mohamed, Abdelrahman and Dupoux, Emmanuel},
  eprint={2203.16502},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2022}
}
```
