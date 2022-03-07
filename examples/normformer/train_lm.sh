#!/usr/bin/env bash
train_common () {
  fairseq-train "$DATA" \
      --combine-val \
      --train-subset train \
      --num-workers 2 \
      --validate-interval-updates 1000 \
      --save-interval-updates 1000 \
      --no-epoch-checkpoints \
      --ddp-backend fully_sharded \
      --memory-efficient-fp16 \
      --fp16-init-scale 4 \
      --checkpoint-activations \
      --arch transformer_lm_gpt \
      --activation-fn gelu \
      --share-decoder-input-output-embed \
      --task language_modeling \
      --sample-break-mode none \
      --tokens-per-sample 2048 \
      --optimizer adam --adam-betas "(0.9, 0.98)" \
      --adam-eps 1e-08 \
      --clip-norm 0.0 \
      --lr-scheduler polynomial_decay \
      --warmup-updates 750 \
      --dropout 0.1 \
      --attention-dropout 0.1 \
      --weight-decay 0.01 \
      --batch-size 16 \
      --update-freq 2 \
      --required-batch-size-multiple 1 \
      --total-num-update 572204 \
      --max-update 572204 \
      --seed 1 \
      --log-format json --log-interval 1 \
      --distributed-world-size 8 --distributed-port 13177 \
        "$@"
}

train_125M () {
  train_common --decoder-layers 12 \
    --decoder-embed-dim 768 \
    --decoder-ffn-embed-dim 3072 \
    --decoder-attention-heads 12 "$@"
}

train_355M () {
  train_common --decoder-layers 24 \
    --decoder-embed-dim 1024\
    --decoder-ffn-embed-dim 4096 \
    --decoder-attention-heads  16 \
    --dropout 0.0 \
    --attention-dropout 0.0 \
    "$@"
}

train_1.3B () {
  train_common --decoder-layers 24 \
    --decoder-embed-dim 2048 \
    --decoder-ffn-embed-dim 8192 \
    --decoder-attention-heads  32 \
    --batch-size 4 \
    --update-freq 16 \
    --total-num-update 286102 \
    --max-update 286102 \
    "$@"
}

train_2.7B () {
    train_common --decoder-layers 32 \
    --decoder-embed-dim 2560 \
    --decoder-ffn-embed-dim 10240 \
    --decoder-attention-heads  32 \
    --batch-size 4 \
    --update-freq 16 \
    --total-num-update 286102 \
    --max-update 286102 \
    "$@"
}
