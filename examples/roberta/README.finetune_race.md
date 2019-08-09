# Finetuning RoBERTa on RACE tasks

### 1) Download the data from RACE website (http://www.cs.cmu.edu/~glai1/data/race/)

### 2) Preprocess RACE data:
```bash
python ./examples/roberta/preprocess_RACE.py <input-dir> <extracted-data-dir>
./examples/roberta/preprocess_RACE.sh <extracted-data-dir> <output-dir>
```

### 3) Fine-tuning on RACE:

```bash
MAX_EPOCHS=5          # epoch number
LR=1e-05              # Peak LR for fixed LR scheduler.
NUM_CLASSES=4
MAX_SENTENCES=2       # batch size
ROBERTA_PATH=/path/to/roberta/model.pt

CUDA_VISIBLE_DEVICES=0 python train.py <race-preprocessed-dir>/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --task sentence_ranking \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_ranking \
    --num-classes $NUM_CLASSES \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler fixed --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --update-freq 8 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
```

**Note:**

a) As contexts in RACE are relatively long, we are using smaller batch size per GPU while increasing update-freq to achieve larger effective batch size.

b) Above cmd-args and hyperparams are tested on one Nvidia `V100` GPU with `32gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq` and reduce `--max-sentences`.

c) The setting in above command is based on our hyperparam search within a fixed search space (for careful comparison across models). You might be able to find better metrics with wider hyperparam search.  
