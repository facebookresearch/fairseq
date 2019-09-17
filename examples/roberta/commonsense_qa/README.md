# Finetuning RoBERTa on Commonsense QA

We follow a similar approach to [finetuning RACE](README.race.md). Specifically
for each question we construct five inputs, one for each of the five candidate
answer choices. Each input is constructed by concatenating the question and
candidate answer. We then encode each input and pass the resulting "[CLS]"
representations through a fully-connected layer to predict the correct answer.
We train with a standard cross-entropy loss.

We also found it helpful to prepend a prefix of `Q:` to the question and `A:` to
the answer. The complete input format is:
```
<s> Q: Where would I not want a fox? </s> A: hen house </s>
```

Our final submission is based on a hyperparameter search over the learning rate
(1e-5, 2e-5, 3e-5), batch size (8, 16), number of training steps (2000, 3000,
4000) and random seed. We selected the model with the best performance on the
development set after 100 trials.

### 1) Download data from the Commonsense QA website (https://www.tau-nlp.org/commonsenseqa)
```bash
bash examples/roberta/commonsense_qa/download_cqa_data.sh
```

### 2) Finetune

```bash
MAX_UPDATES=3000      # Number of training steps.
WARMUP_UPDATES=150    # Linearly increase LR over this many steps.
LR=1e-05              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=16      # Batch size.
SEED=1                # Random seed.
ROBERTA_PATH=/path/to/roberta/model.pt
DATA_DIR=data/CommonsenseQA

# we use the --user-dir option to load the task from
# the examples/roberta/commonsense_qa directory:
FAIRSEQ_PATH=/path/to/fairseq
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/commonsense_qa

CUDA_VISIBLE_DEVICES=0 fairseq-train --fp16 --ddp-backend=no_c10d \
    $DATA_DIR \
    --user-dir $FAIRSEQ_USER_DIR \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task commonsense_qa --init-token 0 --bpe gpt2 \
    --arch roberta_large --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking --num-classes 5 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --max-update $MAX_UPDATES \
    --log-format simple --log-interval 25 \
    --seed $SEED
```

The above command assumes training on 1 GPU with 32GB of RAM. For GPUs with
less memory, decrease `--max-sentences` and increase `--update-freq`
accordingly to compensate.

### 3) Evaluate
```python
import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task
roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'data/CommonsenseQA')
roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
with open('data/CommonsenseQA/valid.jsonl') as h:
    for line in h:
        example = json.loads(line)
        scores = []
        for choice in example['question']['choices']:
            input = roberta.encode(
                'Q: ' + example['question']['stem'],
                'A: ' + choice['text'],
                no_separator=True
            )
            score = roberta.predict('sentence_classification_head', input, return_logits=True)
            scores.append(score)
        pred = torch.cat(scores).argmax()
        answer = ord(example['answerKey']) - ord('A')
        nsamples += 1
        if pred == answer:
            ncorrect += 1

print('Accuracy: ' + str(ncorrect / float(nsamples)))
# Accuracy: 0.7846027846027847
```

The above snippet is not batched, which makes it quite slow. See [instructions
for batched prediction with RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta#batched-prediction).
