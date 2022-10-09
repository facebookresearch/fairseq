# An example of English to Japaneses Simultaneous Translation System

This is an example of training and evaluating a transformer *wait-k* English to Japanese simultaneous text-to-text translation model.

## Data Preparation
This section introduces the data preparation for training and evaluation.
If you only want to evaluate the model, please jump to [Inference & Evaluation](#inference-&-evaluation)

For illustration, we only use the following subsets of the available data from [WMT20 news translation task](http://www.statmt.org/wmt20/translation-task.html), which results in 7,815,391 sentence pairs.
- News Commentary v16
- Wiki Titles v3
- WikiMatrix V1
- Japanese-English Subtitle Corpus
- The Kyoto Free Translation Task Corpus

We use WMT20 development data as development set. Training `transformer_vaswani_wmt_en_de_big` model on such amount of data will result in 17.3 BLEU with greedy search and 19.7 with beam (10) search. Notice that a better performance can be achieved with the full WMT training data.

We use [sentencepiece](https://github.com/google/sentencepiece) toolkit to tokenize the data with a vocabulary size of 32000.
Additionally, we filtered out the sentences longer than 200 words after tokenization.
Assuming the tokenized text data is saved at `${DATA_DIR}`,
we prepare the data binary with the following command.

```bash
fairseq-preprocess \
    --source-lang en --target-lang ja \
    --trainpref ${DATA_DIR}/train \
    --validpref ${DATA_DIR}/dev \
    --testpref ${DATA_DIR}/test \
    --destdir ${WMT20_ENJA_DATA_BIN} \
    --nwordstgt 32000 --nwordssrc 32000 \
    --workers 20
```

## Simultaneous Translation Model Training
To train a wait-k `(k=10)` model.
```bash
fairseq-train ${WMT20_ENJA_DATA_BIN}  \
    --save-dir ${SAVEDIR}
    --simul-type waitk  \
    --waitk-lagging 10  \
    --max-epoch 70  \
    --arch transformer_monotonic_vaswani_wmt_en_de_big \
    --optimizer adam  \
    --adam-betas '(0.9, 0.98)'  \
    --lr-scheduler inverse_sqrt  \
    --warmup-init-lr 1e-07  \
    --warmup-updates 4000  \
    --lr 0.0005  \
    --stop-min-lr 1e-09  \
    --clip-norm 10.0  \
    --dropout 0.3  \
    --weight-decay 0.0  \
    --criterion label_smoothed_cross_entropy  \
    --label-smoothing 0.1  \
    --max-tokens 3584
```
This command is for training on 8 GPUs. Equivalently, the model can be trained on one GPU with `--update-freq 8`.

## Inference & Evaluation
First of all, install [SimulEval](https://github.com/facebookresearch/SimulEval) for evaluation.

```bash
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .
```

The following command is for the evaluation.
Assuming the source and reference files are `${SRC_FILE}` and `${REF_FILE}`, the sentencepiece model file for English is saved at `${SRC_SPM_PATH}`


```bash
simuleval \
    --source ${SRC_FILE} \
    --target ${TGT_FILE} \
    --data-bin ${WMT20_ENJA_DATA_BIN} \
    --sacrebleu-tokenizer ja-mecab \
    --eval-latency-unit char \
    --no-space \
    --src-splitter-type sentencepiecemodel \
    --src-splitter-path ${SRC_SPM_PATH} \
    --agent ${FAIRSEQ}/examples/simultaneous_translation/agents/simul_trans_text_agent_enja.py \
    --model-path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --output ${OUTPUT} \
    --scores
```

The `--data-bin` should be the same in previous sections if you prepare the data from the scratch.
If only for evaluation, a prepared data directory can be found [here](https://dl.fbaipublicfiles.com/simultaneous_translation/wmt20_enja_medium_databin.tgz) and a pretrained checkpoint (wait-k=10 model) can be downloaded from [here](https://dl.fbaipublicfiles.com/simultaneous_translation/wmt20_enja_medium_wait10_ckpt.pt).

The output should look like this:
```bash
{
    "Quality": {
        "BLEU": 11.442253287568398
    },
    "Latency": {
        "AL": 8.6587861866951,
        "AP": 0.7863304776251316,
        "DAL": 9.477850951194764
    }
}
```
The latency is evaluated by characters (`--eval-latency-unit`) on the target side. The latency is evaluated with `sacrebleu` with `MeCab` tokenizer `--sacrebleu-tokenizer ja-mecab`. `--no-space` indicates that do not add space when merging the predicted words.

If `--output ${OUTPUT}` option is used, the detailed log and scores will be stored under the `${OUTPUT}` directory.
