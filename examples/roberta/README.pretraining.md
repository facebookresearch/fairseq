# Pretraining RoBERTa using your own data

This tutorial will walk you through pretraining RoBERTa over your own data.

### 1) Preprocess the data.

Data should be preprocessed following the [language modeling format](/examples/language_model).

We'll use the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
to demonstrate how to preprocess raw text data with the GPT-2 BPE. Of course
this dataset is quite small, so the resulting pretrained model will perform
poorly, but it gives the general idea.

First download the dataset:
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

Next encode it with the GPT-2 BPE:
```bash
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done
```

Finally preprocess/binarize the data using the GPT-2 fairseq dictionary:
```bash
wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref wikitext-103-raw/wiki.train.bpe \
    --validpref wikitext-103-raw/wiki.valid.bpe \
    --testpref wikitext-103-raw/wiki.test.bpe \
    --destdir data-bin/wikitext-103 \
    --workers 60
```

### 2) Train RoBERTa base
```bash
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0005          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16           # Increase the batch size 16x

DATA_DIR=data-bin/wikitext-103

fairseq-train --fp16 $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1
```

The above command assumes training on 8x32GB V100 GPUs. Each GPU uses a batch
size of 16 sequences (`$MAX_SENTENCES`) and accumulates gradients to further
increase the batch size by 16x (`$UPDATE_FREQ`), for a total batch size of 2048
sequences. If you have fewer GPUs or GPUs with less memory you may need to
reduce `$MAX_SENTENCES` and increase `$UPDATE_FREQ` to compensate. Alternatively
if you have more GPUs you can decrease `$UPDATE_FREQ` accordingly to increase
training speed.

Also note that the learning rate and batch size are tightly connected and need
to be adjusted together. We generally recommend increasing the learning rate as
you increase the batch size according to the following table (although it's also
dataset dependent, so don't rely on the following values too closely):

batch size | peak learning rate
---|---
256 | 0.0001
2048 | 0.0005
8192 | 0.0007

### 3) Load your pretrained model
```python
from fairseq.models.roberta import RobertaModel
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'path/to/data')
assert isinstance(roberta.model, torch.nn.Module)
```
