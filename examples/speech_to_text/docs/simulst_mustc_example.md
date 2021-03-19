# Simultaneous Speech Translation (SimulST) on MuST-C

This is a tutorial of training and evaluating a transformer *wait-k* simultaneous model on MUST-C English-Germen Dataset, from [SimulMT to SimulST: Adapting Simultaneous Text Translation to End-to-End Simultaneous Speech Translation](https://www.aclweb.org/anthology/2020.aacl-main.58.pdf).

[MuST-C](https://www.aclweb.org/anthology/N19-1202) is multilingual speech-to-text translation corpus with 8-language translations on English TED talks.

## Data Preparation
This section introduces the data preparation for training and evaluation.
If you only want to evaluate the model, please jump to [Inference & Evaluation](#inference-&-evaluation)

[Download](https://ict.fbk.eu/must-c) and unpack MuST-C data to a path
`${MUSTC_ROOT}/en-${TARGET_LANG_ID}`, then preprocess it with
```bash
# Additional Python packages for S2T data processing/model training
pip install pandas torchaudio sentencepiece

# Generate TSV manifests, features, vocabulary,
# global cepstral and mean estimation,
# and configuration for each language
cd fairseq

python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task asr \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global

python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global
```

## ASR Pretraining
We need a pretrained offline ASR model. Assuming the save directory of the ASR model is `${ASR_SAVE_DIR}`.
The following command (and the subsequent training commands in this tutorial) assume training on 1 GPU (you can also train on 8 GPUs and remove the `--update-freq 8` option).
```
fairseq-train ${MUSTC_ROOT}/en-de \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch convtransformer_espnet --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8
```
A pretrained ASR checkpoint can be downloaded [here](https://dl.fbaipublicfiles.com/simultaneous_translation/must_c_v1_en_de_pretrained_asr)

## Simultaneous Speech Translation Training

### Wait-K with fixed pre-decision module
Fixed pre-decision indicates that the model operate simultaneous policy on the boundaries of fixed chunks.
Here is a example of fixed pre-decision ratio 7 (the simultaneous decision is made every 7 encoder states) and
a wait-3 policy model. Assuming the save directory is `${ST_SAVE_DIR}`
```bash
 fairseq-train ${MUSTC_ROOT}/en-de \
        --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
        --save-dir ${ST_SAVE_DIR} --num-workers 8  \
        --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 4000 --max-update 100000 --max-tokens 40000 --seed 2 \
        --load-pretrained-encoder-from ${ASR_SAVE_DIR}/checkpoint_best.pt \
        --task speech_to_text  \
        --arch convtransformer_simul_trans_espnet  \
        --simul-type waitk_fixed_pre_decision  \
        --waitk-lagging 3 \
        --fixed-pre-decision-ratio 7 \
        --update-freq 8

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
        --fixed-pre-decision-ratio 7 \
        --update-freq 8
```
## Inference & Evaluation
[SimulEval](https://github.com/facebookresearch/SimulEval) is used for evaluation.
The following command is for evaluation.

```
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .

simuleval \
    --agent ${FAIRSEQ}/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py
    --source ${SRC_LIST_OF_AUDIO}
    --target ${TGT_FILE}
    --data-bin ${MUSTC_ROOT}/en-de \
    --config config_st.yaml \
    --model-path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --output ${OUTPUT} \
    --scores
```

The source file `${SRC_LIST_OF_AUDIO}` is a list of paths of audio files. Assuming your audio files stored at `/home/user/data`,
it should look like this

```bash
/home/user/data/audio-1.wav
/home/user/data/audio-2.wav
```

Each line of target file `${TGT_FILE}` is the translation for each audio file input.
```bash
Translation_1
Translation_2
```

The `--data-bin` and `--config` should be the same in previous section if you prepare the data from the scratch.
If only for evaluation, a prepared data directory can be found [here](https://dl.fbaipublicfiles.com/simultaneous_translation/must_c_v1.0_en_de_databin.tgz). It contains
- `spm_unigram10000_st.model`: a sentencepiece model binary.
- `spm_unigram10000_st.txt`: the dictionary file generated by the sentencepiece model.
- `gcmvn.npz`: the binary for global cepstral mean and variance.
- `config_st.yaml`: the config yaml file. It looks like this.
You will need to set the absolute paths for `sentencepiece_model` and `stats_npz_path` if the data directory is downloaded.
```yaml
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: ABS_PATH_TO_SENTENCEPIECE_MODEL
global_cmvn:
  stats_npz_path: ABS_PATH_TO_GCMVN_FILE
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - global_cmvn
  _train:
  - global_cmvn
  - specaugment
vocab_filename: spm_unigram10000_st.txt
```

Notice that once a `--data-bin` is set, the `--config` is the base name of the config yaml, not the full path.

Set `--model-path` to the model checkpoint.
A pretrained checkpoint can be downloaded from [here](https://dl.fbaipublicfiles.com/simultaneous_translation/convtransformer_wait5_pre7), which is a wait-5 model with a pre-decision of 280 ms.

The output should be similar as follow:
```bash
{
    "Quality": {
        "BLEU": 12.79214535384013
    },
    "Latency": {
        "AL": 1669.5778120018108,
        "AL_CA": 2077.9027656104813,
        "AP": 0.7652936521983029,
        "AP_CA": 0.8891561507382866,
        "DAL": 2028.1566141735727,
        "DAL_CA": 2497.336430059716
    }
}
```

If `--output ${OUTPUT}` option is used, the detailed log and scores will be stored under the `${OUTPUT}` directory.


The quality is measured by detokenized BLEU. So make sure that the predicted words sent to the server are detokenized.

The latency metrics are
* Average Proportion
* Average Lagging
* Differentiable Average Lagging

Again they will also be evaluated on detokenized text.
