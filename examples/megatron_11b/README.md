# Megatron-11b

Megatron-11b is a unidirectional language model with `11B` parameters based on [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf). Following the original Megatron work, we trained the model using intra-layer model parallelism with each layer's parameters split across 8 GPUs.

Megatron-11b is trained on the same data and uses the same byte-pair encoding (BPE) as [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf).

## Pre-trained models

Model | Description | # params | # filesize | Download
---|---|---|---|---
`megatron_11b` | megatron_11b unidirectional language model | 11B | 19Gb | [megatron_11b.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/model_parallel/megatron_11b.tar.gz)

#### Architecture:

Param | Value
---|---
embed_dim | 3072
ffn_dim | 3072 * 6
layers | 72
attention heads | 32

#### Training details:

Param | value
---|---
bsz | 512
num_updates | 300,000
peak_lr | 1.5e-04
lr scheduler | inverse_sqrt
clip norm | 0.0


## Example training command (model parallel)

Megatron-11b contains too many parameters to train on a single GPU. Following
the original Megatron work, we adopt an intra-layer model parallel training
approach in which each layer's parameters are split across multiple GPUs and
activations and gradients are communicated during the forward/backward pass,
respectively. We similarly split the loss computation using the
`vocab_parallel_cross_entropy` criterion.

The following training command illustrates how to do model parallel training in
fairseq. We assume that each machine (node) has 8 GPUs among which to split the
model parameters (`--model-parallel-size 8`). If you have access to multiple
nodes, you may combine this with data parallel training by increasing
`--distributed-world-size`.

To train Megatron-11b on a single node:


```bash
fairseq-train <DATA_PATH> \
  --distributed-world-size 8  \
  --memory-efficient-fp16 \
  --num-workers 2 \
  --model-parallel-size 8 \
  --criterion vocab_parallel_cross_entropy \
  --task language_modeling \
  --sample-break-mode none \
  --tokens-per-sample 1024 \
  --arch transformer_lm_megatron_11b \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --lr 0.00015 \
  --warmup-updates 3000 --weight-decay 0.01 \
  --dropout 0.1 --attention-dropout 0.1 \
  --max-sentences 2 \
  --max-update 300000;
```

Note: Above was tested on `DGX-1` box, with `8xV100-32Gb` GPUs.

## Results

**[Wikitext103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)**

Model | vValid perplexity | Test perplexity
---|---|---
`megatron_11b` | 10.64 | 10.54


## Evaluating `megatron_11b` on Wikitext-103

#### 1. Downloading Megatron-11b
```bash
# WARNING: this file is 19GB
wget https://dl.fbaipublicfiles.com/fairseq/models/model_parallel/megatron_11b.tar.gz
tar -xzvf megatron_11b.tar.gz
```

#### 2. Download Wikitext-103
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

#### 3. Detokenize test tokens
Megatron-11b uses a byte-level BPE that expects raw (untokenized) input. Since
the wikitext-103 dataset comes tokenized, we apply a simple detokenization
process to restore the untokenized test set:

```bash
python -m examples.megatron_11b.detok wikitext-103-raw/wiki.test.raw > wikitext-103-raw/wiki.test.detok
```

#### 4. BPE encoding
```bash
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "wikitext-103-raw/wiki.test.detok" \
    --outputs "wikitext-103-raw/wiki.test.bpe" \
    --workers 60;
```

#### 5. Fairseq binarize
```bash
fairseq-preprocess \
    --only-source \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --srcdict megatron_11b/dict.txt \
    --destdir wikitext103-bin;
```

#### 6. Evaluating perplexity.
We can now evaluate perplexity on the test set. Note that because we've modified
the test set (via detokenization and BPE), the perplexity reported by
`fairseq-eval-lm` needs to be renormalized.

Compute unnormalized perplexity:

```bash
DATA_PATH=wikitext103-bin/
fairseq-eval-lm \
  $DATA_PATH \
  --path megatron_11b/model.pt \
  --task language_modeling \
  --gen-subset test \
  --max-sentences 8 \
  --criterion cross_entropy \
  --context-window 992 \
  --distributed-world-size 8 \
  --model-parallel-size 8;
# Expected PPL (unnormalized_ppl): [8.46]
# Note: the eval command needs to run on 8 GPUs for the released model
```
Renormalizing formula:  `2 ^ ( log_2(unnormalized_PPL) * (270847 / 245566))`.
PPL After normalization: `10.54`

To renormalize the perplexity, we must account for the change in token count
after detokenizing and appling BPE. The formula for this is:
`2 ^ ( log_2(unnormalized_PPL) * (new_token_cnt / orig_token_cnt))`

For the wikitext-103 test set, the original token count is `245566` and the
token count after detokenization and applying BPE is `270847`.

The perplexity after renormalization is:
`2 ^ ( log_2(8.46) * (270847 / 245566)) = 10.54`
