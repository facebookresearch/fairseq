# Finetuning RoBERTa on GLUE tasks

### 1) Download the data from GLUE website (https://gluebenchmark.com/tasks) using following commands:
```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```

### 2) Preprocess GLUE task data:
```bash
./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
```
`glue_task_name` is one of the following:
`{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}`
Use `ALL` for preprocessing all the glue tasks.

### 3) Fine-tuning on GLUE task:
Example fine-tuning cmd for `RTE` task
```bash
ROBERTA_PATH=/path/to/roberta/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train -config-dir examples/roberta/config/finetuning --config-name rte \
task.data=RTE-bin checkpoint.restore_file=$ROBERTA_PATH
```

There are additional config files for each of the GLUE tasks in the examples/roberta/config/finetuning directory.

**Note:**

a) Above cmd-args and hyperparams are tested on one Nvidia `V100` GPU with `32gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq` and reduce `--batch-size`.

b) All the settings in above table are suggested settings based on our hyperparam search within a fixed search space (for careful comparison across models). You might be able to find better metrics with wider hyperparam search.

### Inference on GLUE task
After training the model as mentioned in previous step, you can perform inference with checkpoints in `checkpoints/` directory using following python code snippet:

```python
from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='RTE-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('glue_data/RTE/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))

```
