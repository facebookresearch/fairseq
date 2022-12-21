# data2vec 2.0

data2vec 2.0 improves the training efficiency of the original data2vec algorithm. We make the following improvements for efficiency considerations - we forward only the unmasked timesteps through the encoder, we use convolutional decoder and we use multimasking to amortize the compute overhead of the teacher model. You can find details in the paper [Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language](https://arxiv.org/abs/2212.07525) and our [blog post](https://ai.facebook.com/blog/ai-self-supervised-learning-data2vec/).

## Pretrained and finetuned models
### Vision
| Model | Finetuning split | Link
|---|---|---
data2vec ViT-B | No fine-tuning | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet.pt)
data2vec ViT-B | Imagenet-1K  | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet_ft.pt)
data2vec ViT-L | No fine-tuning | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt)
data2vec ViT-L | Imagenet-1K  | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet_ft.pt)
data2vec ViT-H | No fine-tuning | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet.pt)
data2vec ViT-H | Imagenet-1K  | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet_ft.pt)

Vision models only are license under CC-BY-NC.
### Speech

| Model | Finetuning split | Dataset | Link
|---|---|---|---
data2vec Base | No fine-tuning | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_libri.pt)
data2vec Base | 960 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_libri_960h.pt)
data2vec Large | No fine-tuning | [Libri-light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_vox.pt)
data2vec Large | 960 hours | [Libri-light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_vox_960h.pt)

### NLP

Model | Fine-tuning data | Dataset | Link
|---|---|---|---|
data2vec Base | No fine-tuning | Books + Wiki | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec2/nlp_base.pt)

[//]: # (## Data Preparation)

[//]: # ()
[//]: # (### Vision)

[//]: # (add details)

[//]: # (### Speech)

[//]: # (add details)

[//]: # ()
[//]: # (### NLP)

[//]: # (add details)


## Commands to train different models using data2vec 2.0

### Vision

Commands to pretrain different model configurations
```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_images_only_task task.data=/path/to/dir
```

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name large_images_only_task task.data=/path/to/dir
```

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name huge_images14_only_task task.data=/path/to/dir
```

Commands to finetune different model configurations

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/vision/finetuning \
--config-name mae_imagenet_clean task.data=/path/to/dir model.model_path=/path/to/pretrained/model
```

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/vision/finetuning \
--config-name mae_imagenet_large_clean task.data=/path/to/dir model.model_path=/path/to/pretrained/model
```

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/vision/finetuning \
--config-name mae_imagenet_huge_clean task.data=/path/to/dir model.model_path=/path/to/pretrained/model
```

### Speech

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_audio_only_task task.data=/path/to/manifests
```

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name large_audio_only_task task.data=/path/to/manifests
```

Finetuning:

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/wav2vec/config/finetuning --config-name vox_10h \
task.data=/path/to/manifests model.w2v_path=/path/to/pretrained/model common.user_dir=examples/data2vec
```

Replace vox_10h with the right config depending on your model and fine-tuning split. 
See examples/wav2vec/config/finetuning for all available configs.

### NLP

Commands to pretrain
```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_text_only_task task.data=/path/to/file
```

Commands to fine-tune all GLUE tasks
```shell script
$ task=cola  # choose from [cola|qnli|mrpc|rte|sst_2|mnli|qqp|sts_b]
$ lr=1e-5    # sweep [1e-5|2e-5|4e-5|6e-5] for each task
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2/text_finetuning \
--config-name $task task.data=/path/to/file model.model_path=/path/to/pretrained/model "optimization.lr=[${lr}]"
```

# data2vec
  
data2vec is a framework for self-supervised representation learning for images, speech, and text as described in [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language (Baevski et al., 2022)](https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language).  The algorithm uses the same learning mechanism for different modalities. 


## Pre-trained models

### Vision

Code and pre-trained models for data2vec visions can be found [here](https://github.com/facebookresearch/data2vec_vision/tree/main/beit).

### Speech

| Model | Finetuning split | Dataset | Link
|---|---|---|---
data2vec Base | No fine-tuning | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/audio_base_ls.pt)
data2vec Base | 10 minutes | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/audio_base_ls_10m.pt)
data2vec Base | 100 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/audio_base_ls_100h.pt)
data2vec Base | 960 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/audio_base_ls_960h.pt)
data2vec Large | No fine-tuning | [Libri-light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/vox_pretrained.pt)
data2vec Large | 10 minutes | [Libri-light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/vox_10m.pt)
data2vec Large | 100 hours | [Libri-light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/vox_100h.pt)
data2vec Large | 960 hours | [Libri-light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/vox_960h.pt)
---

### NLP

Model | Fine-tuning data | Dataset | Link
|---|---|---|---|
data2vec Base | No fine-tuning | Books + Wiki | [download](https://dl.fbaipublicfiles.com/fairseq/data2vec/nlp_base.pt)

## Training a new speech model with the CLI tools

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate file 10 to 30 seconds in length)

### Prepare training data manifest:

First, install the `soundfile` library:
```shell script
pip install soundfile
```

Next, run:

```shell script
$ python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```

$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.

$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.
To use a pre-defined validation set (like dev-other from librispeech), set to it 0 and then overwrite valid.tsv with a
separately pre-processed manifest file.

### Train a data2vec Base model:

This configuration was used for the base model trained on the Librispeech dataset in the data2vec paper

Note that the input is expected to be single channel, sampled at 16 kHz

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/audio/pretraining \
--config-name base_librispeech task.data=/path/to/manifests common.user_dir=examples/data2vec
```

Note: you can simulate 16 GPUs by using k GPUs and adding command line parameters
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 16/k

### Fine-tune a pre-trained model with CTC:

Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format.
A letter vocabulary can be downloaded [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).
An example [script](../wav2vec/libri_labels.py) that generates labels for the Librispeech dataset from the tsv file produced by wav2vec_manifest.py can be used as follows:

```shell script
split=train
$ python libri_labels.py /path/to/tsv --output-dir /output/dir --output-name $split
```

Fine-tuning on 100h of Librispeech with letter targets:
```shell script
$ fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
    --config-name base_100h common.user_dir=examples/data2vec
```

There are other config files in the config/finetuning directory that can be used to fine-tune on other splits.
You can specify the right config via the `--config-name` parameter.

Decoding with a language model during training requires flashlight [python bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python) (previously called [wav2letter](https://github.com/facebookresearch/wav2letter).
If you want to use a language model, add `+criterion.wer_args='[/path/to/kenlm, /path/to/lexicon, 2, -1]'` to the command line.

### Evaluating a CTC model:

Evaluating a CTC model with a language model requires [flashlight python bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python) (previously called [wav2letter](https://github.com/facebookresearch/wav2letter) to be installed.

Fairseq transformer language model used in the wav2vec 2.0 paper can be obtained from the [wav2letter model repository](https://github.com/facebookresearch/wav2letter/tree/master/recipes/sota/2019).
Be sure to upper-case the language model vocab after downloading it.

Letter dictionary for pre-trained models can be found [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).

Next, run the evaluation command:

```shell script
python examples/speech_recognition/new/infer.py --config-dir examples/speech_recognition/new/conf \
--config-name infer task=audio_finetuning task.data=/path/to/manifests common.user_dir=examples/data2vec \
task.labels=ltr decoding.type=kenlm \
decoding.lmweight=${lmweight} decoding.wordscore=${wordscore} decoding.silweight=${silscore} \
decoding.lexicon=/path/to/lexicon \
decoding.lmpath=/path/to/lm decoding.unique_wer_file=True \
dataset.gen_subset=dev_clean,dev_other,test_clean,test_other \
common_eval.path=/path/to/checkpoint.pt decoding.beam=1500 distributed_training.distributed_world_size=${num_gpus}
```

To get raw numbers, use decoding.type=viterbi and omit the lexicon. To use the transformer language model, use decoding.type=fairseqlm.

## Training a new NLP model with the CLI tools

Please follow the [RoBERTa](../roberta/README.md) instructions to preprocess your data. To train a data2vec model on run:

```shell script
$ python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/text/pretraining \
--config-name base task.data=/path/to/data common.user_dir=examples/data2vec
```

As for speech models, you can simulate 16 gpus by using the update_freq parameter.

### Finetuning data2vec-text on GLUE

Please use a command similar to this:

```shell
$ python fairseq_cli/hydra_train.py -m --config-dir examples/roberta/config/finetuning \
    --config-name $task task.data=$data_path checkpoint.restore_file="${/path/to/pretrained/model.pt}"
```
