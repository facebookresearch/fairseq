# Truncated Backpropagation Through Time (BPTT)

Truncated BPTT is a useful technique for training language models on very long
sequences. Typically a long sequences is split into chunks and a language model
is trained over the chunks sequentially. The LM may condition on previous
chunks, but gradients only flow through the current chunk. This technique was
the basis for the paper: [Transformer-XL: Attentive Language Models Beyond a
Fixed-Length Context](https://arxiv.org/abs/1901.02860), which achieved
state-of-the-art language modeling results at the time of publication.

It is slightly tricky to implement Truncated BPTT efficiently in fairseq, since
we need to iterate over the data sequentially and disable any batch shuffling
logic. The code provided in this example illustrates how to implement Truncated
BPTT in fairseq by overriding ``FairseqTask::get_batch_iterator`` to iterate
over the data sequentially. Crucially, this example supports batching and
multi-GPU (data parallel) training.

##### 0. Setup

First, see the general [language modeling README](README.md) for instructions on
preprocessing the WikiText-103 data.

##### 1. Train a Transformer-XL model on WikiText-103

We will train a 16-layer Transformer-XL model following the [hyperparameters
used in the original
paper](https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/run_wt103_base.sh).

The following command assumes 4 GPUs, so that the total batch size is 60
sequences (15 x 4). Training should take ~24 hours on 4 V100 GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    --user-dir examples/truncated_bptt \
    data-bin/wikitext-103/ \
    --task truncated_bptt_lm --tokens-per-sample 150 \
    --batch-size 15 --max-update 200000 \
    --arch transformer_xl --n-layer 16 --d-model 410 --n-head 10 \
    --d-head 41 --d-inner 2100 --dropout 0.1 --dropatt 0.0 --mem-len 150 \
    --optimizer adam --clip-norm 0.25 \
    --lr-scheduler cosine --warmup-updates 0 --min-lr 0.0 --lr 0.00025  \
    --log-format json --log-interval 25 \
    --fp16
```

If training on a single GPU, set `--update-freq=4` to accumulate 4x gradients
and simulate training on 4 GPUs.

##### 2. Evaluate

```bash
fairseq-eval-lm data-bin/wikitext-103/ \
    --path checkpoints/checkpoint_best.pt \
    --user-dir examples/truncated_bptt/ \
    --task truncated_bptt_lm \
    --batch-size 1 --required-batch-size-multiple 1 \
    --model-overrides '{"mem_len":640,"clamp_len":400,"same_length":True}' \
    --tokens-per-sample 64
# ... | INFO | fairseq_cli.eval_lm | num. model params: 151123537
# ... | INFO | fairseq_cli.eval_lm | Evaluated 245569 tokens in 83.1s (2956.82 tokens/s)
# ... | INFO | fairseq_cli.eval_lm | Loss (base 2): 4.5668, Perplexity: 23.70
# Compare to 24.0 test perplexity from the paper
```

*Note:* During training the model saw 150 tokens of context
(``--tokens-per-sample=150``) and 150 extra memory tokens (``--mem-len=150``).
During evaluation we measure perplexity on sequences of 64 tokens
(``--tokens-per-sample=64``) and increase the memory length
(``--model-overrides='{"mem_len":640}'``). These settings match the evaluation
settings from [the original
paper](https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/run_wt103_base.sh).
