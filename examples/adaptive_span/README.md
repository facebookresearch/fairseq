# Adaptive Span

Adaptive Span is a novel self-attention mechanism that can learn its optimal
attention span. This allows us to extend significantly the maximum context size
used in Transformer, while maintaining control over their memory footprint
and computational time. It uses the Truncated BPTT technique for training,
as in [transformerXL](https://github.com/pytorch/fairseq/blob/master/examples/truncated_bptt/README.md).

Adaptive Span was introduced by paper:
[Adaptive Attention Span in Transformers](https://arxiv.org/abs/1905.07799),
which achieved state-of-the-art language modeling results at the time of publication.

We manage to reproduce their result in fairseq and keep most of the
[original implementation](https://github.com/facebookresearch/adaptive-span) untouched.
You can refer to the their sweep file as well if any combination of hyperparameter is not clear.

##### 0. Setup

First you need to process the Enwik8 dataset, we use the pre-tokenized dataset
from [adaptive span paper](https://github.com/facebookresearch/adaptive-span/blob/master/get_data.sh).
You can download the dataset, and then run:
```bash
fairseq-preprocess --only-source --trainpref ~/data/enwik8/train.txt \
    --validpref ~/data/enwik8/valid.txt --testpref ~/data/enwik8/test.txt \
    --destdir ~/data/enwik8/data-bin/ --joined-dictionary --workers 20
```

##### 1. Train a Adaptive Span model on Enwik8

We will train a 12-layer Adaptive Span model following the [hyperparameters
used in the original
paper](https://github.com/facebookresearch/adaptive-span/blob/master/experiments/enwik8.sh).

The following command assumes 4 GPUs, so that the total batch size is 64
sequences (4 x 16). Training should take 2-3 days on 4 V100 GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    --user-dir examples/adaptive_span \
    --data  ~/data/enwik8/data-bin/ \
    --fp16 --fp16-no-flatten-grads --max-update 600000 \
    --task truncated_bptt_lm --tokens-per-sample 512 --arch adaptive_span \
    --n-layer 12 --d-model 512 --n-head 8 --d-inner 2048 --dropout 0.3 \
    --attn-span 8192 --optimizer adagrad_with_grad_clip --adagrad-clip 0.03 \
    --validate-interval-updates 1000 \
    --lr-scheduler fixed --warmup-updates 32000 --batch-size-valid 32 \
    --lr 0.07 --criterion adaptive_span_loss --batch-size 16 --update-freq 1 \
    --seed 2 --log-format json --log-interval 25 --aux-loss-scaler 5e-07
```
This should land around 1.05 on validation, 1.03 on test. You can lower the
--aux-loss-scaler for better performance (longer span). It gives ~0.03 bpc
improvement to the transformerXL baseline here.
If training on a single GPU, set `--update-freq=4` to accumulate 4x gradients
and simulate training on 4 GPUs.
You can also reproduce the transformerXL result on enwik8 using this code base.
It should land around 1.06 on test,matching the [original paper](https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/run_enwik8_base.sh).
You can try by
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    --user-dir examples/truncated_bptt \
    ~/data/enwik8/data-bin/ \
    --task truncated_bptt_lm  --fp16 --max-update 400000 \
    --tokens-per-sample 512 --arch transformer_xl --n-layer 12 \
    --d-model 512 --n-head 8 --d-head 64 --d-inner 2048 --dropout 0.1 \
    --dropatt 0.0 --mem-len 512 --optimizer adam --clip-norm 0.25 \
    --lr-scheduler cosine --warmup-updates 0 \
    --lr 0.0 --lr 0.00025 --batch-size 15 \
    --update-freq 1 --seed 2 --log-format json --log-interval 25 \
    --fp16
```

##### 2. Evaluate
For Adaptive Span:
```bash
fairseq-eval-lm ~/data/enwik8/data-bin/ --path model/checkpoint_best.pt \
 --user-dir examples/adaptive_span \
 --task truncated_bptt_lm --batch-size 8 --tokens-per-sample 512 --gen-subset test
```
For Transformer-XL evaluation:
```bash
fairseq-eval-lm ~/data/enwik8/data-bin/ --path model/checkpoint_best.pt \
    --user-dir examples/truncated_bptt/ --task truncated_bptt_lm --batch-size 8 \
    --tokens-per-sample 80 \
    --model-overrides '{"mem_len":2100,"clamp_len":820,"same_length":True}' \
    --gen-subset valid
```

*Note:* During training the model saw 512 tokens of context
(``--tokens-per-sample=512``), with batch size 8. These settings match the evaluation
settings from [the original
paper](https://github.com/facebookresearch/adaptive-span/blob/master/experiments/enwik8.sh).
