# Simultaneous Machine Translation

This directory contains the code for the paper [Monotonic Multihead Attention](https://openreview.net/forum?id=Hyg96gBKPS)

## Prepare Data

[Please follow the instructions to download and preprocess the WMT'15 En-De dataset.](https://github.com/pytorch/fairseq/tree/simulastsharedtask/examples/translation#prepare-wmt14en2desh)

## Training

- MMA-IL

```shell
fairseq-train \
    data-bin/wmt15_en_de_32k \
    --simul-type infinite_lookback \
    --user-dir $FAIRSEQ/example/simultaneous_translation \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --latency-weight-avg  0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en save_dir_key=lambda \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 3584
```

- MMA-H

```shell
fairseq-train \
    data-bin/wmt15_en_de_32k \
    --simul-type hard_aligned \
    --user-dir $FAIRSEQ/example/simultaneous_translation \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --latency-weight-var  0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en save_dir_key=lambda \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 3584
```

- wait-k

```shell
fairseq-train \
    data-bin/wmt15_en_de_32k \
    --simul-type wait-k \
    --waitk-lagging 3 \
    --user-dir $FAIRSEQ/example/simultaneous_translation \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en save_dir_key=lambda \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 3584
```


## Evaluation

More details on evaluation can be found [here](https://github.com/pytorch/fairseq/blob/simulastsharedtask/examples/simultaneous_translation/docs/evaluation.md)

### Start the server

```shell
python ./eval/server.py \
    --src-file $SRC_FILE \
    --ref-file $TGT_FILE
```

### Run the client

```shell
python ./evaluate.py \
    --data-bin data-bin/wmt15_en_de_32k \
    --model-path ./checkpoints/checkpoint_best.pt
    --scores --output $RESULT_DIR
```

### Run evaluation locally without server

```shell
python ./eval/evaluate.py
    --local \
    --src-file $SRC_FILE \
    --tgt-file $TGT_FILE \
    --data-bin data-bin/wmt15_en_de_32k \
    --model-path ./checkpoints/checkpoint_best.pt \
    --scores --output $RESULT_DIR
```
