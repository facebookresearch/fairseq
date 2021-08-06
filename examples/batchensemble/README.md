### Command Line

Uses the included `scripts/train.sh`

```bash
./train.sh \
  checkpoints/be_LL \
  "3" \
  --user-dir examples/batchensemble/src \
  --task multilingual_translation_be \
  --batch-ensemble-root 0 \
  --arch batch_ensemble_multilingual_transformer \
  --share-decoders --share-decoder-input-output-embed \
  --memory-efficient-fp16 --tensorboard-logdir checkpoints/be_LL &
```

```bash
./eval.sh \
  ./fairseq \
  ./2017-01-mted-test \
  checkpoints/be_LL_linear \
  "3" \
  --user-dir fairseq/examples/batchensemble/src --task multilingual_translation_be \
  --batch-size 64 --memory-efficient-fp16
```

./eval.sh \
  ./fairseq \
  ./2017-01-mted-test \
  checkpoints/be_LL_linear \
  "3" \
  --user-dir fairseq/examples/batchensemble/src --task multilingual_translation \
  --batch-size 64 --memory-efficient-fp16
