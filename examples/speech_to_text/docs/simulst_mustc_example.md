# Simultaneous Speech Translation (SimulST) on MuST-C

[MuST-C](https://www.aclweb.org/anthology/N19-1202) is multilingual speech-to-text translation corpus with 8-language translations on English TED talks.

## Data Preparation & ASR
Please follow the steps in offline [speech-to-text](../mustc_example.md) translation for data preparation and ASR pretraining.

## Training

#### Wait-K(K=3) with fixed pre-decision module
```
 fairseq-train ${MUSTC_ROOT}/en-de \
        --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
        --save-dir ${ST_SAVE_DIR} --num-workers 8  \
        --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 4000 --max-update 100000 --max-tokens 40000 --seed 2 \
        --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} \
        --task speech_to_text  \
        --arch convtransformer_simul_trans_espnet  \
        --simul-type waitk_fixed_pre_decision  \
        --waitk-lagging 3 \
        --fixed-pre-decision-ratio 7
```
#### Monotonic multihead attention with fixed pre-decision module
```
 fairseq-train ${MUSTC_ROOT}/en-de \
        --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
        --save-dir ${ST_SAVE_DIR} --num-workers 8  \
        --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --warmup-updates 4000 --max-update 100000 --max-tokens 40000 --seed 2 \
        --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} \
        --task speech_to_text  \
        --criterion latency_augmented_label_smoothed_cross_entropy \
        --latency-weight-avg 0.1 \
        --arch convtransformer_simul_trans_espnet  \
        --simul-type infinite_lookback_fixed_pre_decision  \
        --fixed-pre-decision-ratio 7
```
## Inference & Evaluation
[SimulEval](https://github.com/facebookresearch/SimulEval) is used for evaluation.
```
simuleval \
    --agent ${FAIRSEQ}/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py
    --src-file ${SRC_LIST_OF_AUDIO}
    --tgt-file ${TGT_FILE}
    --data-bin ${MUSTC_ROOT}/en-de \
    --model-path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --tgt-splitter-type SentencePieceModel \
    --tgt-splitter-path ${MUSTC_ROOT}/en-de/spm.model \
    --scores
```
