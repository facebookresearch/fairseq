# Fine-tuning details

For each task (GLUE and PAWS), we perform hyperparam search for each model, and report the mean and standard deviation across 5 seeds of the best model. First, get the datasets following the instructions in [RoBERTa fine-tuning README](../roberta/README.glue.md). Alternatively, you can use [huggingface datasets](https://huggingface.co/docs/datasets/) to get the task data:

```python
from datasets import load_dataset
import pandas as pd
from pathlib import Path

key2file = {
"paws": {
        "loc": "paws_data",
        "columns": ["id", "sentence1", "sentence2", "label"],
        "train": "train.tsv",
        "validation": "dev.tsv",
        "test": "test.tsv"
  }
}

task_data = load_dataset("paws", "labeled_final")
task_config = key2file["paws"]
save_path = Path(task_config["loc"])
save_path.mkdir(exist_ok=True, parents=True)
for key, fl in task_config.items():
    if key in ["loc", "columns"]:
        continue
    print(f"Reading {key}")
    columns = task_config["columns"]
    df = pd.DataFrame(task_data[key])
    print(df.columns)
    df = df[columns]
    print(f"Got {len(df)} records")
    save_loc = save_path / fl
    print(f"Saving to : {save_loc}")
    df.to_csv(save_loc, sep="\t", header=None, index=None)

```

- Preprocess using RoBERTa GLUE preprocessing script, while keeping in mind the column numbers for `sentence1`, `sentence2` and `label` (which is 0,1,2 if you save the data according to the above example.)
- Then, fine-tuning is performed similarly to RoBERTa (for example, in case of RTE):

```bash
TOTAL_NUM_UPDATES=30875  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=1852      # 6 percent of the number of updates
LR=2e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size.
SHUFFLED_ROBERTA_PATH=/path/to/shuffled_roberta/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train RTE-bin/ \
    --restore-file $SHUFFLED_ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
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
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
```

- `TOTAL_NUM_UPDATES` is computed based on the `--batch_size` value and the dataset size.
- `WARMUP_UPDATES` is computed as 6% of `TOTAL_NUM_UPDATES`
- Best hyperparam of `--lr` and `--batch_size` is reported below:

## `--lr`

|     | name         |   RTE |  MRPC | SST-2 |  CoLA |   QQP |  QNLI |  MNLI |  PAWS |
| --: | :----------- | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
|   0 | original     | 2e-05 | 2e-05 | 1e-05 | 2e-05 | 1e-05 | 1e-05 | 1e-05 | 2e-05 |
|   1 | n_1          | 2e-05 | 1e-05 | 1e-05 | 1e-05 | 3e-05 | 1e-05 | 2e-05 | 2e-05 |
|   2 | n_2          | 2e-05 | 2e-05 | 1e-05 | 1e-05 | 2e-05 | 1e-05 | 1e-05 | 3e-05 |
|   3 | n_3          | 3e-05 | 1e-05 | 2e-05 | 2e-05 | 3e-05 | 1e-05 | 1e-05 | 2e-05 |
|   4 | n_4          | 3e-05 | 1e-05 | 2e-05 | 2e-05 | 2e-05 | 1e-05 | 1e-05 | 2e-05 |
|   5 | r512         | 1e-05 | 3e-05 | 2e-05 | 2e-05 | 3e-05 | 2e-05 | 3e-05 | 2e-05 |
|   6 | rand_corpus  | 2e-05 | 1e-05 | 3e-05 | 1e-05 | 3e-05 | 3e-05 | 3e-05 | 2e-05 |
|   7 | rand_uniform | 2e-05 | 1e-05 | 3e-05 | 2e-05 | 3e-05 | 3e-05 | 3e-05 | 1e-05 |
|   8 | rand_init    | 1e-05 | 1e-05 | 3e-05 | 1e-05 | 1e-05 | 1e-05 | 2e-05 | 1e-05 |
|   9 | no_pos       | 1e-05 | 3e-05 | 2e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 | 1e-05 |

## `--batch_size`

|     | name         | RTE | MRPC | SST-2 | CoLA | QQP | QNLI | MNLI | PAWS |
| --: | :----------- | --: | ---: | ----: | ---: | --: | ---: | ---: | ---: |
|   0 | orig         |  16 |   16 |    32 |   16 |  16 |   32 |   32 |   16 |
|   1 | n_1          |  32 |   32 |    16 |   32 |  32 |   16 |   32 |   16 |
|   2 | n_2          |  32 |   16 |    32 |   16 |  32 |   32 |   16 |   32 |
|   3 | n_3          |  32 |   32 |    16 |   32 |  32 |   16 |   32 |   32 |
|   4 | n_4          |  32 |   16 |    32 |   16 |  32 |   32 |   32 |   32 |
|   5 | r512         |  32 |   16 |    16 |   32 |  32 |   16 |   16 |   16 |
|   6 | rand_corpus  |  16 |   16 |    16 |   16 |  32 |   16 |   16 |   32 |
|   7 | rand_uniform |  16 |   32 |    16 |   16 |  32 |   16 |   16 |   16 |
|   8 | rand_init    |  16 |   16 |    32 |   16 |  16 |   16 |   32 |   16 |
|   9 | no_pos       |  16 |   32 |    16 |   16 |  32 |   16 |   16 |   16 |

- Perform inference similar to RoBERTa as well:

```python
from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='PAWS-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('paws_data/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[0], tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))

```
