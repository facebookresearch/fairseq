# wav2vec

Example to train a wav2vec model as described in [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](https://arxiv.org/abs/1904.05862).

## Pre-trained models

Description | Parameters | Dataset | Model
---|---:|---|---
Wav2Vec large <br> ([(Schneider et al., 2019)](https://arxiv.org/abs/1904.05862)) | 32.5M | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt)

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

Given a directory containing wav files to be used for pretraining (we recommend splitting each file into separate file 10 to 30 seconds in length)

### Prepare training data manifest:

```
$ python scripts/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext wav
```

### Train a wav2vec model:

```
$ python train.py /manifest/path --save-dir /model/path --num-workers 6 --fp16 --max-update 400000 --save-interval 1 --no-epoch-checkpoints \
--arch wav2vec --task audio_pretraining --lr 1e-06 --min-lr 1e-09 --optimizer adam --max-lr 0.005 --lr-scheduler cosine \
--conv-feature-layers [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)] \
--conv-aggregator-layers [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)] \
--skip-connections-agg --residual-scale 0.5 --log-compression --warmup-updates 500 --warmup-init-lr 1e-07 --criterion binary_cross_entropy --num-negatives 10 \
--max-sample-size 150000 --max-tokens 1500000 ---skip-invalid-size-inputs-valid-test
```

### Extract embeddings from the downstream task data:

```
$ PYTHONPATH /path/to/fairseq python examples/wav2vec/wav2vec_featurize.py --input /path/to/task/waves --output /path/to/output \
--model /model/path/checkpoint_best.pt --split train valid test
```

# vq-wav2vec

Example to train a vq-wav2vec model as described in [vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations (Baevski et al., 2019)](https://arxiv.org/abs/1910.05453).

## Pre-trained models

Description | Parameters | Dataset | Model
---|---:|---|---
vq-wav2vec Gumbel <br> ([(Baevski et al., 2019)](https://arxiv.org/abs/1910.05453)) | 34.1M | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt)
vq-wav2vec K-means <br> ([(Baevski et al., 2019)](https://arxiv.org/abs/1910.05453)) | 33.0M | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt)
Roberta on K-means codes <br> ([(Baevski et al., 2019)](https://arxiv.org/abs/1910.05453)) | 123.6M | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/wav2vec/bert_kmeans.tar)

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
--warmup-init-lr 1e-07 --criterion binary_cross_entropy --num-negatives 10 --max-sample-size 150000 \
--max-tokens 300000 --cross-sample-negatives 0 --update-freq 1 --seed 2 --skip-invalid-size-inputs-valid-test
```

for k-means training, set vq-type with "kmeans" and add --loss-weights [1] argument. Pre-trained models were trained on 16 GPUs.

### Tokenize audio data (e.g. for BERT training):

```
$ PYTHONPATH /path/to/fairseq python examples/wav2vec/vq-wav2vec_featurize.py --data-dir /manifest/path --output-dir /path/to/output \
--checkpoint /model/path/checkpoint_best.pt --split train valid test --extension tsv
```