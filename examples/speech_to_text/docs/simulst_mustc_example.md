# Simultaneous Speech Translation (SimulST) on MuST-C

This is an instruction of training and evaluating a transformer *wait-k* simultaneous model on MUST-C English-Germen Dataset, from [SimulMT to SimulST: Adapting Simultaneous Text Translation to End-to-End Simultaneous Speech Translation](https://www.aclweb.org/anthology/2020.aacl-main.58.pdf).

[MuST-C](https://www.aclweb.org/anthology/N19-1202) is multilingual speech-to-text translation corpus with 8-language translations on English TED talks.

## Data Preparation
[Download](https://ict.fbk.eu/must-c) and unpack MuST-C data to a path
`${MUSTC_ROOT}/en-${TARGET_LANG_ID}`, then preprocess it with
```bash
# Additional Python packages for S2T data processing/model training
pip install pandas torchaudio sentencepiece

# Generate TSV manifests, features, vocabulary,
# global cepstral and mean estimation,
# and configuration for each language
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global
python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 10000
  --cmvn-type global
```

## ASR Pretraining
We just need a pretrained offline ASR model
```
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch convtransformer_espnet --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```

## Simultaneous Speech Translation Training

### Wait-K with fixed pre-decision module
Fixed pre-decision indicates that the model operate simultaneous policy on the boundaries of fixed chunks.
Here is a example of fixed pre-decision ratio 7 (the simultaneous decision is made every 7 encoder states) and
a wait-3 policy model
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
### Monotonic multihead attention with fixed pre-decision module
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
The source file is a list of paths of audio files,
while target file is the corresponding translations.
```
pip install simuleval

simuleval \
    --agent examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py
    --src-file ${SRC_LIST_OF_AUDIO}
    --tgt-file ${TGT_FILE}
    --data-bin ${MUSTC_ROOT}/en-de \
    --model-path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --tgt-splitter-type SentencePieceModel \
    --tgt-splitter-path ${MUSTC_ROOT}/en-de/spm.model \
    --scores
```

A pretrained checkpoint can be downloaded from [here](https://dl.fbaipublicfiles.com/simultaneous_translation/convtransformer_wait5_pre7), which is a wait-5 model with a pre-decision of 280 ms. The databin (containing dictionary, gcmvn file and sentencepiece model) can be found [here](https://dl.fbaipublicfiles.com/simultaneous_translation/must_c_v1.0_en_de_databin).

The quality is measured by detokenized BLEU. So make sure that the predicted words sent to the server are detokenized.

The latency metrics are
* Average Proportion
* Average Lagging
* Differentiable Average Lagging
Again they will also be evaluated on detokenized text.
