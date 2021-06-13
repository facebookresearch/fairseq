# HuBERT

## Pre-trained and fine-tuned (ASR) models
Model | Pretraining Data | Finetuning Dataset | Model
|---|---|---|---
HuBERT Base (~95M params) | [Librispeech](http://www.openslr.org/12) 960 hr | No finetuning (Pretrained Model) | [download](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)
HuBERT Large (~316M params) | [Libri-Light](https://github.com/facebookresearch/libri-light) 60k hr | No finetuning (Pretrained Model) | [download](https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt)
HuBERT Extra Large (~1B params) | [Libri-Light](https://github.com/facebookresearch/libri-light) 60k hr |  No finetuning (Pretrained Model) | [download](https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt)
HuBERT Large | [Libri-Light](https://github.com/facebookresearch/libri-light) 60k hr | [Librispeech](http://www.openslr.org/12) 960 hr | [download](https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k_finetune_ls960.pt)
HuBERT Extra Large | [Libri-Light](https://github.com/facebookresearch/libri-light) 60k hr | [Librispeech](http://www.openslr.org/12) 960 hr | [download](https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k_finetune_ls960.pt)


## Train a new model

### Data preparation

Follow the steps in `./simple_kmeans` to create:
- `{train,valid}.tsv` waveform list files
- `{train,valid}.km` frame-aligned pseudo label files.
The `label_rate` is the same as the feature frame rate used for clustering,
which is 100Hz for MFCC features and 50Hz for HuBERT features by default.

### Pre-train a HuBERT model

Suppose `{train,valid}.tsv` are saved at `/path/to/data`, `{train,valid}.km`
are saved at `/path/to/labels`, and the label rate is 100Hz.

To train a base model (12 layer transformer), run:
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech \
  task.data=/path/to/data task.label_dir=/path/to/labels model.label_rate=100
```

### Fine-tune a HuBERT model with a CTC loss

Suppose `{train,valid}.tsv` are saved at `/path/to/data`, and their
corresponding character transcripts `{train,valid}.ltr` are saved at
`/path/to/trans`.

To fine-tune a pre-trained HuBERT model at `/path/to/checkpoint`, run
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/finetune \
  --config-name base_10h \
  task.data=/path/to/data task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```

### Decode a HuBERT model

Suppose the `test.tsv` and `test.ltr` are the waveform list and transcripts of
the split to be decoded, saved at `/path/to/data`, and the fine-tuned model is
saved at `/path/to/checkpoint`. We support three decoding modes:
- Viterbi decoding: greedy decoding without a language model
- KenLM decoding: decoding with an arpa-format KenLM n-gram language model
- Fairseq-LM deocding: decoding with a Fairseq neural language model


#### Viterbi decoding

`task.normalize` needs to be consistent with the value used during fine-tuning.
Decoding results will be saved at
`/path/to/experiment/directory/decode/viterbi/test`.

```sh
$ python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/hubert/config/decode \
  --config-name infer_viterbi \
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
  --config-dir /path/to/fairseq-py/examples/hubert/config/decode \
  --config-name infer_kenlm \
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

To decode with a Fairseq LM, use `--config-name infer_fsqlm` instead, and
change the path of lexicon and LM accordingly.
