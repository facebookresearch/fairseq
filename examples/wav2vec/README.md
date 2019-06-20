# wav2vec

Example to train a wav2vec model as described in [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](https://arxiv.org/abs/1904.05862).

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
$ PYTHONPATH /path/to/fairseq python scripts/wav2vec_featurize.py --input /path/to/task/waves --output /path/to/output \
--model /model/path/checkpoint_best.pt --split train valid test
```
