# wav2vec 2.0

wav2vec 2.0 learns speech representations on unlabeled data as described in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477).

## Pre-trained models

Model | Finetuning split | Dataset | Model
|---|---|---|---
Wav2Vec 2.0 Base | No finetuning | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)
Wav2Vec 2.0 Base | 10 minutes | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_10m.pt)
Wav2Vec 2.0 Base | 100 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_100h.pt)
Wav2Vec 2.0 Base | 960 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt)
Wav2Vec 2.0 Large | No finetuning | [Librispeech](http://www.openslr.org/12)  | [download](https//dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt)
Wav2Vec 2.0 Large | 10 minutes | [Librispeech](http://www.openslr.org/12)  | [download](https//dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_10m.pt)
Wav2Vec 2.0 Large | 100 hours | [Librispeech](http://www.openslr.org/12)  | [download](https//dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_100h.pt)
Wav2Vec 2.0 Large | 960 hours | [Librispeech](http://www.openslr.org/12)  | [download](https//dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_960h.pt)
Wav2Vec 2.0 Large (LV-60) | No finetuning | [Libri-Light](https://github.com/facebookresearch/libri-light) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox.pt)
Wav2Vec 2.0 Large (LV-60) | 10 minutes | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_10m.pt)
Wav2Vec 2.0 Large (LV-60) | 100 hours | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_100h.pt)
Wav2Vec 2.0 Large (LV-60) | 960 hours | [Libri-Light](https://github.com/facebookresearch/libri-light) + [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec2_vox_960h.pt)

## Training a new model with the CLI tools

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

### Train a wav2vec 2.0 base model:

This configuration was used for the base model trained on the Librispeech dataset in the wav2vec 2.0 paper

Note that this was tested with pytorch 1.4.0 and the input is expected to be single channel, sampled at 16 kHz

```shell script
$ python train.py --distributed-world-size 64 --distributed-port $PORT /manifest/path \
--save-dir /model/path --fp16 --num-workers 6 --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 --latent-vars 320 \
--latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
--lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
--max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d
```

Note: you can simulate 64 GPUs by using k GPUs and setting --update-freq 64/k

### Train a wav2vec 2.0 large model:

This configuration was used for the large model trained on the Libri-light dataset in the wav2vec 2.0 paper

```shell script
$ python train.py --distributed-world-size 128 --distributed-port $PORT /manifest/path \
--save-dir /model/path --fp16 --num-workers 6 --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
--log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
--conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 768 --latent-vars 320 \
--latent-groups 2 --latent-temp '(2.0,0.1,0.999995)' --infonce --optimizer adam \
--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 600000 \
--lr 0.0003 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
--encoder-layerdrop 0.0 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.03 \
--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --encoder-layers 24 --encoder-embed-dim 1024 \
--encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --num-negatives 100 --cross-sample-negatives 0 \
--max-sample-size 320000 --min-sample-size 32000 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.01 \
--max-tokens 1200000 --max-update 600000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d
```

Note: you can simulate 128 GPUs by using k GPUs and setting --update-freq 128/k

### Fine-tune a pre-trained model with CTC:

Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format.
A letter vocabulary can be downloaded [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).
An example [script](libri_labels.py) that generates labels for the Librispeech dataset from the tsv file produced by wav2vec_manifest.py can be used as follows:

```shell script
split=train
$ python libri_labels.py /path/to/tsv --output-dir /output/dir --output-name $split
```

Fine-tuning on 100h of Librispeech with letter targets:
```shell script
valid_subset=dev_other
python train.py --distributed-world-size 24 --distributed-port $PORT /path/to/training_data --save-dir /model/path --fp16 \
--wer-args '("/path/to/lm/4-gram.bin","/path/to/lexicon",2,-1)' \
--post-process letter --valid-subset $valid_subset --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 4 \
--max-update 80000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path /path/to/pretrained/model \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 \
--decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --log-format json --log-interval 500 --ddp-backend no_c10d
```

Note: you can simulate 24 GPUs by using k GPUs and setting --update-freq 24/k

Decoding with a language model during training requires wav2letter [python bindings](https://github.com/facebookresearch/wav2letter/wiki/Building-Python-bindings).
Alternatively, simply omit the --wer-args flag.

### Evaluating a CTC model:

Evaluating a CTC model with a language model requires wav2letter [python bindings](https://github.com/facebookresearch/wav2letter/wiki/Building-Python-bindings) to be installed.

Fairseq transformer language model used in the wav2vec 2.0 paper can be obtained from the [wav2letter model repository](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/sota/2019).
Be sure to upper-case the language model vocab after downloading it.

Letter dictionary for pre-trained models can be found [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).

Next, run the evaluation command:

```shell script
$subset=dev_other
python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw --task audio_pretraining \
--nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --w2l-decoder kenlm \
--lm-model /path/to/kenlm.bin --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
--post-process letter
```

To get raw numbers, use --w2l-decoder viterbi and omit the lexicon. To use the transformer language model, use --w2l-decoder fairseqlm.

# wav2vec

Example to train a wav2vec model as described in [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](https://arxiv.org/abs/1904.05862).

## Pre-trained models

Description | Dataset | Model
---|---|---
Wav2Vec large | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt)

#### Example usage:
```python
import torch
from fairseq.models.wav2vec import Wav2VecModel

cp = torch.load('/path/to/wav2vec.pt')
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
c = model.feature_aggregator(z)
```

## Training a new model with the CLI tools

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate files 10 to 30 seconds in length)

### Prepare training data manifest:

```
$ python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext wav
```

### Train a wav2vec model:

```
$ python train.py /manifest/path --save-dir /model/path --num-workers 6 --fp16 --max-update 400000 --save-interval 1 --no-epoch-checkpoints \
--arch wav2vec --task audio_pretraining --lr 1e-06 --min-lr 1e-09 --optimizer adam --max-lr 0.005 --lr-scheduler cosine \
--conv-feature-layers [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)] \
--conv-aggregator-layers [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)] \
--skip-connections-agg --residual-scale 0.5 --log-compression --warmup-updates 500 --warmup-init-lr 1e-07 --criterion wav2vec --num-negatives 10 \
--max-sample-size 150000 --max-tokens 1500000 --skip-invalid-size-inputs-valid-test
```

### Extract embeddings from the downstream task data:

```
$ PYTHONPATH=/path/to/fairseq python examples/wav2vec/wav2vec_featurize.py --input /path/to/task/waves --output /path/to/output \
--model /model/path/checkpoint_best.pt --split train valid test
```

# vq-wav2vec

Example to train a vq-wav2vec model as described in [vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations (Baevski et al., 2019)](https://arxiv.org/abs/1910.05453).

These models are also used in [Effectiveness of self-supervised pre-training for speech recognition (Baevski et al., 2019)](https://arxiv.org/abs/1911.03912).

## Pre-trained models

Description | Dataset | Model
---|---|---
vq-wav2vec Gumbel | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt)
vq-wav2vec K-means | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt)
Roberta on K-means codes | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/bert_kmeans.tar)

#### Example usage:
```python
import torch
from fairseq.models.wav2vec import Wav2VecModel

cp = torch.load('/path/to/vq-wav2vec.pt')
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

wav_input_16khz = torch.randn(1,10000)
z = model.feature_extractor(wav_input_16khz)
_, idxs = model.vector_quantizer.forward_idx(z)
print(idxs.shape) # output: torch.Size([1, 60, 2]), 60 timesteps with 2 indexes corresponding to 2 groups in the model
```

## Training a new model with the CLI tools

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate file 10 to 30 seconds in length)

### Prepare training data manifest:

```
$ python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext wav
```

### Train a gumbel vq-wav2vec model:

```
$ python train.py /manifest/path --save-dir /model/path --num-workers 6 --fp16 --max-update 400000 \
--save-interval 1 --no-epoch-checkpoints --arch wav2vec --task audio_pretraining --lr 1e-06 --min-lr 1e-09 \
--optimizer adam --max-lr 1e-05 --lr-scheduler cosine \
--conv-feature-layers [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)] \
--conv-aggregator-layers [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)] \
--activation gelu --offset auto --skip-connections-agg --residual-scale 0.5 \
--log-keys ["prob_perplexity","code_perplexity","temp"] --vq-type gumbel --vq-groups 2 --vq-depth 2 \
--combine-groups --vq-vars 320 --vq-temp (2,0.5,0.999995) --prediction-steps 12 --warmup-updates 1000 \
--warmup-init-lr 1e-07 --criterion wav2vec --num-negatives 10 --max-sample-size 150000 \
--max-tokens 300000 --cross-sample-negatives 0 --update-freq 1 --seed 2 --skip-invalid-size-inputs-valid-test
```

for k-means training, set vq-type with "kmeans" and add --loss-weights [1] argument. Pre-trained models were trained on 16 GPUs.

### Tokenize audio data (e.g. for BERT training):

```
$ PYTHONPATH=/path/to/fairseq python examples/wav2vec/vq-wav2vec_featurize.py --data-dir /manifest/path --output-dir /path/to/output \
--checkpoint /model/path/checkpoint_best.pt --split train valid test --extension tsv
```
