# Fine-tuning BART on CNN-Dailymail summarization task

### 1) Download the CNN and Daily Mail data and preprocess it into data files with non-tokenized cased samples.

Follow the instructions [here](https://github.com/abisee/cnn-dailymail) to download the original CNN and Daily Mail datasets. To preprocess the data, refer to the pointers in [this issue](https://github.com/pytorch/fairseq/issues/1391) or check out the code [here](https://github.com/artmatsak/cnn-dailymail).

Follow the instructions [here](https://github.com/EdinburghNLP/XSum) to download the original Extreme Summarization datasets, or check out the code [here](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset), Please keep the raw dataset and make sure no tokenization nor BPE on the dataset.

### 2) BPE preprocess:

```bash
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASK=cnn_dm
for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
```

### 3) Binarize dataset:
```bash
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
```

### 4) Fine-tuning on CNN-DM summarization task:
Example fine-tuning CNN-DM
```bash
TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/path/to/bart/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train cnn_dm-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
```
Above is expected to run on `1` node with `8 32gb-V100`.
Expected training time is about `5 hours`. Training time can be reduced with distributed training on `4` nodes and `--update-freq 1`.

Use TOTAL_NUM_UPDATES=15000 UPDATE_FREQ=2 for Xsum task

### Inference for CNN-DM test data using above trained checkpoint.
After training the model as mentioned in previous step, you can perform inference with checkpoints in `checkpoints/` directory using `eval_cnn.py`, for example

```bash
cp data-bin/cnn_dm/dict.source.txt  checkpoints/
python examples/bart/summarize.py \
  --model-dir checkpoints \
  --model-file checkpoint_best.pt \
  --src cnn_dm/test.source \
  --out cnn_dm/test.hypo
```
For XSUM, which uses beam=6, lenpen=1.0, max_len_b=60, min_len=10:
```bash
cp data-bin/cnn_dm/dict.source.txt  checkpoints/
python examples/bart/summarize.py \
  --model-dir checkpoints \
  --model-file checkpoint_best.pt \
  --src cnn_dm/test.source \
  --out cnn_dm/test.hypo \
  --xsum-kwargs
```
