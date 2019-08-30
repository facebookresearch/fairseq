# Finetuning RoBERTa on RACE tasks

### 1) Download the data from RACE website (http://www.cs.cmu.edu/~glai1/data/race/)

### 2) Preprocess RACE data:
```bash
python ./examples/roberta/preprocess_RACE.py --input-dir <input-dir> --output-dir <extracted-data-dir>
./examples/roberta/preprocess_RACE.sh <extracted-data-dir> <output-dir>
```

### 3) Fine-tuning on RACE:

```bash
MAX_EPOCH=5           # Number of training epochs.
LR=1e-05              # Peak LR for fixed LR scheduler.
NUM_CLASSES=4
MAX_SENTENCES=1       # Batch size per GPU.
UPDATE_FREQ=8         # Accumulate gradients to simulate training on 8 GPUs.
DATA_DIR=/path/to/race-output-dir
ROBERTA_PATH=/path/to/roberta/model.pt

CUDA_VISIBLE_DEVICES=0,1 fairseq-train $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sentence_ranking \
    --num-classes $NUM_CLASSES \
    --init-token 0 --separator-token 2 \
    --max-option-length 128 \
    --max-positions 512 \
    --truncate-sequence \
    --arch roberta_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler fixed --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-sentences $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-epoch $MAX_EPOCH
```

**Note:**

a) As contexts in RACE are relatively long, we are using smaller batch size per GPU while increasing update-freq to achieve larger effective batch size.

b) Above cmd-args and hyperparams are tested on one Nvidia `V100` GPU with `32gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq` and reduce `--max-sentences`.

c) The setting in above command is based on our hyperparam search within a fixed search space (for careful comparison across models). You might be able to find better metrics with wider hyperparam search.  
