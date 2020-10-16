[Better Fine-Tuning by Reducing Representational Collapse](https://arxiv.org/abs/2008.03156)
=====================
This repo contains the code to replicate all experiments from the _Better Fine-Tuning by Reducing Representational Collapse_ paper excluding the probing results.

The R3F sentence prediction criterion is registered as `sentence_prediction_r3f` while the label smoothing version of it is implemented as `label_smoothed_cross_entropy_r3f`. The R4F version of the sentence prediction criterion can be achieved by applying spectral norm to the classification head via the `--spectral-norm-classification-head` parameter.

## Hyper-parameters
Our methods introduce 3 new hyper-parameters; `--eps` which sets the standard deviation or range of the distribution we're sampling from, `--r3f-lambda` which controls the combining of logistic loss and noisy KL loss and `--noise-type` which controls which parametric distribution we use ('normal', 'uniform').

For example to run R3F on RTE from GLUE

```
TOTAL_NUM_UPDATES=3120
WARMUP_UPDATES=187
LR=1e-05
NUM_CLASSES=2
MAX_SENTENCES=8        # Batch size.
ROBERTA_PATH=/path/to/roberta/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train RTE-bin \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction_r3f \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --noise-type uniform --r3f-lambda 0.7 \
    --user-dir examples/rxf;
```

## Citation
```bibtex
@article{aghajanyan2020better,
  title={Better Fine-Tuning by Reducing Representational Collapse},
  author={Aghajanyan, Armen and Shrivastava, Akshat and Gupta, Anchit and Goyal, Naman and Zettlemoyer, Luke and Gupta, Sonal},
  journal={arXiv preprint arXiv:2008.03156},
  year={2020}
}
```
