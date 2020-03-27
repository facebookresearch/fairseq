# Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2019)
This page contains information for how to train models with Quantization Noise.

Check out our blog post [here](link_to_blog_post) and read the paper [here](link_to_paper).

Looking for pretrained models? They will be added shortly.
Looking for code to train vision models? We are working on open sourcing our code as part of ClassyVision. Please check back.

## Citation:
To be added shortly

## Description and Example Usage

Training a model with Quant-Noise improves the performance in subsequent inference-time quantization by training models to be robust. This technique is useful for both scalar and vector quantization methods, as well as multiple domains.

### QuickStart

To train a model with Quant-Noise, add the following flags:
```
--quant-noise 0.1 --quant-noise-block-size 8
```
We recommend training with 0.05 to 0.2 Quant-Noise, a value that worked well in our experiments. For the block-size, we recommend training with block-size 8.

Quant-Noise can also be combined with LayerDrop (see [here](https://github.com/pytorch/fairseq/tree/master/examples/layerdrop)) to add its pruning effect to the quantized model and make the model even smaller. We recommend training with LayerDrop 0.1 or 0.2.

To quantize a model, use `train_quantizer.py`.

### Detailed Description

Quantization with Quant-Noise proceeds in two steps. First, a model must be trained with quant-noise. Second, the model must be quantized.

**Step 1**: Training a model with quant-noise.

We currently support training Transformer based models, such as sequence-to-sequence, language models, and BERT architectures. The `quant_noise` function [here](https://github.com/pytorch/fairseq/tree/master/fairseq/modules/quant_noise.py) wraps a module. It splits a weight matrix into blocks and applies random dropout to these blocks.

There are two parameters: `--quant-noise` and `--quant-noise-block-size`. `quant-noise` controls how much dropout is applied to the blocks of the weight matrix. `quant-noise-block-size` controls the size of the weight matrix blocks.

In the Transformer architectures, quant-noise is applied to the input and output embeddings, the attention, and the FFN.

**Step 2**: Quantizing a model.

We currently support two kinds of quantization: scalar quantization such as int4 and int8, and vector quantization in the form of product quantization. We implement an improved version of product quantization from Stock et al described [here](https://arxiv.org/abs/1907.05686).

[add more details to this section!]

### Looking to reproduce the NLP results in the paper?

1. To train RoBERTa + QuantNoise, we followed this setting [here](https://github.com/pytorch/fairseq/tree/master/examples/roberta). The following command can be used to train a RoBERTa Base + QuantNoise model on bookscorpus + wikipedia dataset:

```bash
TOTAL_UPDATES=125000
WARMUP_UPDATES=10000
PEAK_LR=0.0005
TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512
MAX_SENTENCES=16
UPDATE_FREQ=2
DATA_DIR=/path/to/data/here

python train.py $DATA_DIR \
    --task masked_lm --criterion masked_lm --arch roberta_base \
    --sample-break-mode complete \
    --tokens-per-sample $TOKENS_PER_SAMPLE --max-positions $MAX_POSITIONS \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES \
    --update-freq $UPDATE_FREQ --max-update $TOTAL_UPDATES \
    --save-dir checkpoint/roberta \
    --ddp-backend no_c10d --encoder-layerdrop 0.2 \
    --quant-noise 0.2 --quant-noise-block-size 8 --untie-weights-roberta
```

To finetune RoBERTa + QuantNoise, we followed this setting [here](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md). The following command can be used to finetune a RoBERTa Base + QuantNoise model on the RTE dataset:

```bash
TOTAL_NUM_UPDATES=2036  
WARMUP_UPDATES=122      
LR=2e-05               
NUM_CLASSES=2
MAX_SENTENCES=16        
ROBERTA_PATH=/path/to/roberta_quantnoise/model.pt

python train.py /path/to/rte/data/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric
    --quant-noise 0.2 --quant-noise-block-size 8;
```

2. To train Language Models on Wikitext-103, we followed this setting [here](https://github.com/pytorch/fairseq/tree/master/examples/language_model). The following command can be used to train a Transformer + QuantNoise model on Wikitext-103:

```bash
python train.py --task language_modeling /path/to/wikitext-103/data \
    --save-dir checkpoints/transformer_wikitext-103 \
    --adaptive-input --adaptive-input-cutoff 20000,60000 --adaptive-input-factor 4 \
    --adaptive-softmax-cutoff 20000,60000 --adaptive-softmax-dropout 0.2 --adaptive-softmax-factor 4.0 \
    --tie-adaptive-proj --tie-adaptive-weights \
    --arch transformer_lm_gbw \
    --attention-dropout 0.1 --dropout 0.2 --relu-dropout 0.1 \
    --clip-norm 0.1 --criterion adaptive_loss \
    --ddp-backend no_c10d \
    --decoder-attention-heads 8 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 --decoder-input-dim 1024 \
    --decoder-layers 16 --decoder-normalize-before --decoder-output-dim 1024 \
    --lr 0.0001 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 --max-lr 1.0 --t-mult 2.0 \
    --max-tokens 3072 --tokens-per-sample 3072 --momentum 0.99 --optimizer nag \
    --sample-break-mode none --update-freq 3 \
    --warmup-init-lr 1e-07 --warmup-updates 16000 \
    --weight-decay 0 --seed 1 --min-lr 1e-09 \
    --quant-noise 0.05 --quant-noise-block-size 8
```

To evaluate this model, note you need to use the `eval.py` script. The following command can be used to evaluate:

```bash
python eval_lm.py /path/to/wikitext-103/data --path /path/to/model/checkpoint \
    --sample-break-mode complete \
    --max-tokens 3072 \
    --context-window 2560 \
    --softmax-batch 1024 \
    --gen-subset valid
```
and change the `--gen-subset` to `test` if you would like to evaluate on the test set instead.


### Looking to reproduce the Vision results in the paper?

We are working on open sourcing our code as part of ClassyVision. Please check back.


## Having an issue or have a question?

Please open an issue in this repository with the details of your question. Thanks!
