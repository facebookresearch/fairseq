- arxiv link
### Data Preprocessing
Follow [these instructions](https://github.com/pytorch/fairseq/blob/main/examples/language_model/README.md#1-preprocess-the-data) for your dataset.

### Training Commands
```bash
common_train_command () {
    fairseq-train  $DATA \ 
    --save-dir normformer_125M \
    --combine-val --train-subset train --num-workers 2 --validate-interval-updates 1000 --save-interval-updates 10000 \
    --keep-interval-updates 5 --no-epoch-checkpoints --no-best-checkpoints --ddp-backend fully_sharded --memory-efficient-fp16 \
    --fp16-init-scale 4 --checkpoint-activations --arch transformer_lm_gpt --activation-fn gelu --share-decoder-input-output-embed \
    --task language_modeling --sample-break-mode none --tokens-per-sample 2048 --optimizer adam --adam-betas '"'"'(0.9, 0.98)'"'"' --adam-eps 1e-08 \
    --seed 1 --log-format json --log-interval 10 \
    --clip-norm 0.0 --lr-scheduler polynomial_decay "$@"
}
# 125M
common_train_command \
    --lr 0.003 --total-num-update 572204 --warmup-updates 750 \ 
    --dropout 0.1 --attention-dropout 0.1 \
    --decoder-layers 12 --decoder-embed-dim 768 --decoder-ffn-embed-dim 3072 --decoder-attention-heads 12 \
    --weight-decay 0.01 --batch-size 4 --required-batch-size-multiple 1 --update-freq 1 --max-update 572204 \
    --scale-attn --scale-fc --scale-heads --scale-resids \
    --distributed-world-size 64 --distributed-port 18252
#355M


#1.3B

```


### Other hparam notes
