# Discriminative Reranking for Neural Machine Translation
https://aclanthology.org/2021.acl-long.563/

This folder contains source code for training DrNMT, a discriminatively trained reranker for neural machine translation.

## Data preparation
1. Follow the instructions under `examples/translation` to build a base MT model. Prepare three files, one with source sentences, one with ground truth target sentences, and one with hypotheses generated from the base MT model. Each line in the file contains one sentence in raw text (i.e. no sentencepiece, etc.). Below is an example of the files with _N_ hypotheses for each source sentence.

```
# Example of the source sentence file: (The file should contain L lines.)

source_sentence_1
source_sentence_2
source_sentence_3
...
source_sentence_L

# Example of the target sentence file: (The file should contain L lines.)

target_sentence_1
target_sentence_2
target_sentence_3
...
target_sentence_L

# Example of the hypotheses file: (The file should contain L*N lines.)

source_sentence_1_hypo_1
source_sentence_1_hypo_2
...
source_sentence_1_hypo_N
source_sentence_2_hypo_1
...
source_sentence_2_hypo_N
...
source_sentence_L_hypo_1
...
source_sentence_L_hypo_N
```

2. Download the [XLMR model](https://github.com/fairinternal/fairseq-py/tree/master/examples/xlmr#pre-trained-models).
```
wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz
tar zxvf xlmr.base.tar.gz

# The folder should contain dict.txt, model.pt and sentencepiece.bpe.model.
```

3. Prepare scores and BPE data.
* `N`: Number of hypotheses per each source sentence. We use 50 in the paper.
* `SPLIT`: Name of the data split, i.e. train, valid, test. Use split_name, split_name1, split_name2, ..., if there are multiple datasets for a split, e.g. train, train1, valid, valid1.
* `NUM_SHARDS`: Number of shards. Set this to 1 for non-train splits.
* `METRIC`: The metric for DrNMT to optimize for. We support either `bleu` or `ter`.
```
# For each data split, e.g. train, valid, test, etc., run the following:

SOURCE_FILE=/path/to/source_sentence_file
TARGET_FILE=/path/to/target_sentence_file
HYPO_FILE=/path/to/hypo_file
XLMR_DIR=/path/to/xlmr
OUTPUT_DIR=/path/to/output

python scripts/prep_data.py \
    --input-source ${SOURCE_FILE} \
    --input-target ${TARGET_FILE} \
    --input-hypo ${HYPO_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --split $SPLIT
    --beam $N \
    --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
    --metric $METRIC \
    --num-shards ${NUM_SHARDS}

# The script will create ${OUTPUT_DIR}/$METRIC with ${NUM_SHARDS} splits.
# Under split*/input_src, split*/input_tgt and split*/$METRIC, there will be $SPLIT.bpe and $SPLIT.$METRIC files, respectively.

```

4. Pre-process the data into fairseq format.
```
# use comma to separate if there are more than one train or valid set
for suffix in src tgt ; do
    fairseq-preprocess --only-source \
        --trainpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/train.bpe \
        --validpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/valid.bpe \
        --destdir ${OUTPUT_DIR}/$METRIC/split1/input_${suffix} \
        --workers 60 \
        --srcdict ${XLMR_DIR}/dict.txt
done

for i in `seq 2 ${NUM_SHARDS}`; do
    for suffix in src tgt ; do
        fairseq-preprocess --only-source \
            --trainpref ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix}/train.bpe \
            --destdir ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix} \
            --workers 60 \
            --srcdict ${XLMR_DIR}/dict.txt

        ln -s ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/valid* ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix}/.
    done

    ln -s ${OUTPUT_DIR}/$METRIC/split1/$METRIC/valid* ${OUTPUT_DIR}/$METRIC/split${i}/$METRIC/.
done
```

## Training

```
EXP_DIR=/path/to/exp

# An example of training the model with the config for De-En experiment in the paper.
# The config uses 16 GPUs and 50 hypotheses.
# For training with fewer number of GPUs, set
# distributed_training.distributed_world_size=k +optimization.update_freq='[x]' where x = 16/k
# For training with fewer number of hypotheses, set
# task.mt_beam=N dataset.batch_size=N dataset.required_batch_size_multiple=N

fairseq-hydra-train -m \
    --config-dir config/ --config-name deen \
    task.data=${OUTPUT_DIR}/$METRIC/split1/ \
    task.num_data_splits=${NUM_SHARDS} \
    model.pretrained_model=${XLMR_DIR}/model.pt \
    common.user_dir=${FAIRSEQ_ROOT}/examples/discriminative_reranking_nmt \
    checkpoint.save_dir=${EXP_DIR}

```

## Inference & scoring
Perform DrNMT reranking (fw + reranker score)
1. Tune weights on valid sets.
```
# genrate N hypotheses with the base MT model (fw score)
VALID_SOURCE_FILE=/path/to/source_sentences # one sentence per line, converted to the sentencepiece used by the base MT model
VALID_TARGET_FILE=/path/to/target_sentences # one sentence per line in raw text, i.e. no sentencepiece and tokenization
MT_MODEL=/path/to/mt_model
MT_DATA_PATH=/path/to/mt_data

cat ${VALID_SOURCE_FILE} | \
    fairseq-interactive ${MT_DATA_PATH} \
    --max-tokens 4000 --buffer-size 16 \
    --num-workers 32 --path ${MT_MODEL} \
    --beam $N --nbest $N \
    --post-process sentencepiece &> valid-hypo.out

# replace "bleu" with "ter" to optimize for TER
python drnmt_rerank.py \
    ${OUTPUT_DIR}/$METRIC/split1/ \
    --path ${EXP_DIR}/checkpoint_best.pt \
    --in-text valid-hypo.out \
    --results-path ${EXP_DIR} \
    --gen-subset valid \
    --target-text ${VALID_TARGET_FILE} \
    --user-dir ${FAIRSEQ_ROOT}/examples/discriminative_reranking_nmt \
    --bpe sentencepiece \
    --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
    --beam $N \
    --batch-size $N \
    --metric bleu \
    --tune

```

2. Apply best weights on test sets
```
# genrate N hypotheses with the base MT model (fw score)
TEST_SOURCE_FILE=/path/to/source_sentences  # one sentence per line, converted to the sentencepiece used by the base MT model

cat ${TEST_SOURCE_FILE} | \
    fairseq-interactive ${MT_DATA_PATH} \
    --max-tokens 4000 --buffer-size 16 \
    --num-workers 32 --path ${MT_MODEL} \
    --beam $N --nbest $N \
    --post-process sentencepiece &> test-hypo.out

# replace "bleu" with "ter" to evaluate TER
# Add --target-text for evaluating BLEU/TER,
# otherwise the script will only generate the hypotheses with the highest scores only.
python drnmt_rerank.py \
    ${OUTPUT_DIR}/$METRIC/split1/ \
    --path ${EXP_DIR}/checkpoint_best.pt \
    --in-text test-hypo.out \
    --results-path ${EXP_DIR} \
    --gen-subset test \
    --user-dir ${FAIRSEQ_ROOT}/examples/discriminative_reranking_nmt \
    --bpe sentencepiece \
    --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
    --beam $N \
    --batch-size $N \
    --metric bleu \
    --fw-weight ${BEST_FW_WEIGHT} \
    --lenpen ${BEST_LENPEN}
```

## Citation
```bibtex
@inproceedings{lee2021discriminative,
  title={Discriminative Reranking for Neural Machine Translation},
  author={Lee, Ann and Auli, Michael and Ranzato, Marc'Aurelio},
  booktitle={ACL},
  year={2021}
}
```
