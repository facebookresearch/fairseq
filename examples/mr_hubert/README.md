# MR-HuBERT

## Pre-trained and fine-tuned (ASR) models
Model | Pretraining Data | Finetuning Dataset | Model | Quantizer
|---|---|---|---|---

## Load a model
```
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

## Train a new model

### Data preparation

Follow the steps in `./simple_kmeans` to create:
- `{train,valid}.tsv` waveform list files with length information
```
/path/to/your/audio/files
file1.wav\t160000
file2.wav\t154600
...
filen.wav\t54362
```
- `{train,valid}.km` frame-aligned pseudo label files (the order is the same as wavefiles in the tsv file).
```
44 44 44 48 48 962 962 962 962 962 962 962 962 967 967 967 967 967 967 967 967 370 852 370 ... 18 18 745 745
44 44 44 48 48 962 962 962 147 147 147 147 147 147 147 147 147 147 147 147 176 176 271 271 ... 27 27 745 745
...
44 44 44 48 962 962 962 962 962 962 377 377 377 77 77 852 696 694 433 578 578 82 740 622 ... 27 27 745 745
```
- `dict.km.txt` a dummy dictionary (first column is id, the second is dummy one)
```
0 1
1 1
2 1
...
999 1
```

The `label_rate` is the same as the feature frame rate used for clustering,
which is 100Hz for MFCC features and 50Hz for HuBERT features by default.

### Pre-train a MR-HuBERT model

Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.km`
are saved at `/path/to/labels`, and the label rate is 100Hz.

To train a base model (12 layer transformer), run:
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/mr_hubert/config/pretrain \
  --config-name mrhubert_base_librispeech \
  task.data=/path/to/data task.label_dir=/path/to/labels \
  task.labels='["km"]' model.label_rate=100 \
  task.label_rate_ratios='[1, 2]' \
```

Please see sample pre-training scripts `train.sh` for an example script.

### Fine-tune a MR-HuBERT model with a CTC loss

Suppose `{train,valid}.tsv` are saved at `/path/to/data`, and their
corresponding character transcripts `{train,valid}.ltr` are saved at
`/path/to/trans`. A typical ltr file is with the same order of tsv waveform files as 
```
HOW | ARE | YOU
...
THANK | YOU
```

To fine-tune a pre-trained MR-HuBERT model at `/path/to/checkpoint`, run
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/mr_hubert/config/finetune \
  --config-name base_10h \
  task.data=/path/to/data task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```

Please see sample pre-training scripts `finetune.sh` for an example script.

### Decode a MR-HuBERT model

Suppose the `test.tsv` and `test.ltr` are the waveform list and transcripts of
the split to be decoded, saved at `/path/to/data`, and the fine-tuned model is
saved at `/path/to/checkpoint`. 


We support three decoding modes:
- Viterbi decoding: greedy decoding without a language model
- KenLM decoding: decoding with an arpa-format KenLM n-gram language model
- Fairseq-LM deocding: decoding with a Fairseq neural language model (not fully tested)


#### Viterbi decoding

`task.normalize` needs to be consistent with the value used during fine-tuning.
Decoding results will be saved at
`/path/to/experiment/directory/decode/viterbi/test`.

```sh
$ python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/mr_hubert/config/decode \
  --config-name infer \
  task.data=/path/to/data \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint
  dataset.gen_subset=test \
```

#### KenLM / Fairseq-LM decoding

Suppose the pronunciation lexicon and the n-gram LM are saved at
`/path/to/lexicon` and `/path/to/arpa`, respectively. Decoding results will be
saved at `/path/to/experiment/directory/decode/kenlm/test`.

```sh
$ python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/mr_hubert/config/decode \
  --config-name infer_lm \
  task.data=/path/to/data \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint
  dataset.gen_subset=test \
  decoding.decoder.lexicon=/path/to/lexicon \
  decoding.decoder.lmpath=/path/to/arpa
```

The command above uses the default decoding hyperparameter, which can be found
in `examples/speech_recognition/hydra/decoder.py`. These parameters can be
configured from the command line. For example, to search with a beam size of
500, we can append the command above with `decoding.decoder.beam=500`.
Important parameters include:
- decoding.decoder.beam
- decoding.decoder.beamthreshold
- decoding.decoder.lmweight
- decoding.decoder.wordscore
- decoding.decoder.silweight

To decode with a Fairseq LM, you may check the usage examples in wav2vec2 or hubert examples.

Please see sample pre-training scripts `decode.sh`  for an example script.
