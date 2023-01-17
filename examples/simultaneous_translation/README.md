# IWSLT 2023 Shared Task On Simultaneous Translation

This document provides an instruction to reproduce the baseline systems and prepare docker image for submissions for IWSLT 2023 Shared Task on Simultaneous Translation. The instruction provides

- A speech-to-text (S2T) test-time [Wait-K](https://aclanthology.org/P19-1289/) baseline. The model in test-time Wait-K S2T model is trained on MuST-C v2.0 data, with pretrained wav2vec encoder and mBART decoder
- [To be updated] A speech-to-speech (S2S), with an incremental text-to-speech (S2T) synthesis system on top of the test-time Wait-K S2T system.
- Preparation for the docker image for submission

## Data Preparation

This section covers the data preparation required for training.

If you are only interested in model inference / evaluation,
please jump to the [Inference & Evaluation](#inference--evaluation) section.

[Download](https://ict.fbk.eu/must-c-release-v2-0/) and unpack the MuST-C 2.0 data to the path
`${MUSTC_ROOT}/en-${TARGET_LANG}`. Then run the following commands below to preprocess the data.

`${TARGET_LANG}` can be `de`,`zh` and `ja`.

```bash
# additional python packages for S2T data processing / model training
pip install pandas torchaudio sentencepiece

# generate TSV manifests
cd fairseq

python examples/speech_to_text/prep_mustc_v2_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 10000
```

## Pretrained Encoder & Decoder

This section covers open-sourced pretrained encoders and decoders.

If you already have your own pretrained encoder / decoder, please jump to the next section.

For pretrained encoder, we used a [wav2vec 2.0 model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt) opensourced by the [original wav2vec 2.0 paper](https://arxiv.org/abs/2006.11477). Download and extract this model to `${MUSTC_ROOT}/en-${TARGET_LANG}/wav2vec_small_960h.pt`

For pretrained decoder, we used an [mBART model](https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz) opensourced by the [original mBART paper](https://arxiv.org/abs/2001.08210). Download and extract the model to `${MUSTC_ROOT}/en-${TARGET_LANG}/model.pt`, the dict to `${MUSTC_ROOT}/en-${TARGET_LANG}/dict.txt` and the sentencepiece model to `${MUSTC_ROOT}/en-${TARGET_LANG}/sentence.bpe.model`

If using the above mBART model, update `${MUSTC_ROOT}/en-${TARGET_LANG}/config_st.yaml` to look like

```bash
audio_root: ${MUSTC_ROOT}
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: sentence.bpe.model
standardize_audio: true
use_audio_input: true
vocab_filename: dict.txt
```

## Training

This section covers training an offline ST model.

Set `${ST_SAVE_DIR}` to be the save directory of the resulting ST model. This train command assumes that you are training on `1 GPU`, so please adjust the `update-freq` value accordingly.

```bash
 fairseq-train ${MUSTC_ROOT}/en-${TARGET_LANG} \
        --save-dir ${ST_SAVE_DIR} \
        --seed 1 --num-workers 1 --fp16 \
        --task speech_to_text  \
        --arch xm_transformer  \
        --config-yaml config_st.yaml \
        --train-subset train_st --valid-subset dev_st \
        --optimizer adam --lr-scheduler inverse_sqrt --lr 0.0001 --update-freq 64 \
        --clip-norm 10.0 --activation-fn gelu --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
        --warmup-updates 2000 --max-update 30000 \
        --max-tokens 1024 --max-tokens-valid 1024 --max-target-positions 1024 --max-source-positions 1024 \
        --share-decoder-input-output-embed \
        --finetune-w2v-params all --finetune-decoder-params encoder_attn,layer_norm,self_attn \
        --w2v-path ${MUSTC_ROOT}/en-${TARGET_LANG}/wav2vec_small_960h.pt \
        --load-pretrained-decoder-from ${MUSTC_ROOT}/en-${TARGET_LANG}/model.pt \
        --decoder-normalize-before \
        --adaptor-proj \
        --layerdrop 0.1 --decoder-layerdrop 0.1 --adaptor-layerdrop 0.05 \
        --apply-mask --mask-prob 0.1 --mask-length 5
```

## Inference & Evaluation

This section covers simultaneous evaluation using the Wait-K policy.

First, install [SimulEval](https://github.com/facebookresearch/SimulEval) for evaluation.

```
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .
```

Then download the [MuST-C v2.0 tst-COMMON data (segmented)](https://dl.fbaipublicfiles.com/simultaneous_translation/iwslt2023/must-c_v2.0_tst-COMMON.tgz).

After the [Training](#training), collect necessary files and copy them in one directory named `system` for evaluation

- The checkpoint, named as `checkpoint.pt`
- The sentencepiece model from`${MUSTC_ROOT}/en-${TARGET_LANG}/sentence.bpe.model` (only for en-de)
- The config yaml file from `${MUSTC_ROOT}/en-${TARGET_LANG}/config.yaml`
- The dictionary from `${MUSTC_ROOT}/en-${TARGET_LANG}/dict.txt`
- A simultaneous S2T agent script from [s2t_agent.py (en)](), [s2t_agent.py (ja)](), [s2t_agent.py (zh)]()
- A wav2vec config file from [wav2vec_small.yaml](./wav2vec_small.yaml).

`ls system` should give following output:

```bash
> ls system
checkpoint.pt  config.yaml  dict.txt  s2t_agent.py  sentence.bpe.model  wav2vec_small.yaml
```

You can download the prepared systems here [en-de](https://dl.fbaipublicfiles.com/simultaneous_translation/iwslt2023/baseline/en-de), [en-zh](https://dl.fbaipublicfiles.com/simultaneous_translation/iwslt2023/baseline/en-ze), [en-ja](https://dl.fbaipublicfiles.com/simultaneous_translation/iwslt2023/baseline/en-ja)

Running the following commands to reproduce the baseline results.

```bash
# En-De
k=4
step=2
simuleval \
    --agent system/s2t_agent.py \
    --source en-de/wavs.txt --target en-de/refs.txt \
    --checkpoint system/checkpoint.pt \
    --sentencepiece-model system/sentence.bpe.model \
    --config-yaml system/config.yaml \
    --wav2vec-yaml system/wav2vec_small.yaml \
    --source-segment-size 160 \
    --waitk-lagging ${k} \
    --fixed-pre-decision-ratio ${step} \
    --device cuda:0 \
    --output ${OUTPUT_DIR}-de

# En-Zh
k=5
step=2
simuleval \
    --agent system/s2t_agent.py \
    --source en-zh/wavs.txt --target en-zh/refs.txt \
    --checkpoint system/checkpoint.pt \
    --config-yaml system/config.yaml \
    --wav2vec-yaml system/wav2vec_small.yaml \
    --source-segment-size 160 \
    --waitk-lagging ${k} \
    --fixed-pre-decision-ratio ${step} \
    --eval-latency-unit char \
    --sacrebleu-tokenizer ja-mecab \
    --filtered-tokens '▁' \
    --device cuda:0 \
    --output ${OUTPUT_DIR}-zh

# En-Ja
k=4
step=2
simuleval \
    --agent system/s2t_agent.py \
    --source en-ja/wavs.txt --target en-ja/refs.txt \
    --checkpoint system/checkpoint.pt \
    --config-yaml system/config.yaml \
    --wav2vec-yaml system/wav2vec_small.yaml \
    --source-segment-size 160 \
    --waitk-lagging ${k} \
    --fixed-pre-decision-ratio ${step} \
    --eval-latency-unit char \
    --sacrebleu-tokenizer zh \
    --filtered-tokens '▁' \
    --device cuda:0 \
    --output ${OUTPUT_DIR}-ja
```

The results on `tst-COMMON` are:

```bash
> cat ${OUTPUT_DIR}-de/scores.tsv
BLEU    LAAL    AL      AP      DAL
14.251  2100.747        1965.509        0.732   1886.226

> cat ${OUTPUT_DIR}-zh/scores.tsv
BLEU    LAAL    AL      AP      DAL
14.157  1935.869        1776.798        0.748   1884.306

> cat ${OUTPUT_DIR}-ja/scores.tsv
BLEU    LAAL    AL      AP      DAL
6.627   1966.418        1817.957        0.706   1765.576
```

## Preparing submission

Before starting this section, make sure that you are able to reproduce the results in [Inference & Evaluation](#inference--evaluation) section.
We will use English to German system as an example in this section.
Given the `system` directory for evaluation, we add one more script `start_agent.sh`, which contains

```
#!/bin/bash
set -e
step=2
k=4

simuleval --standalone \
    --remote-port 2023 \
    --agent system/s2t_agent.py \
    --checkpoint system/checkpoint.pt \
    --sentencepiece-model system/sentence.bpe.model \
    --config-yaml system/config.yaml \
    --wav2vec-yaml system/wav2vec_small.yaml \
    --source-segment-size 160 --waitk-lagging ${k} --fixed-pre-decision-ratio ${step} \
    --device cuda:0
```

Then build the docker image.

```bash
# Build
docker build -t iwslt2023_s2t_en-de --no-cache .

# Save the image as tar
docker save iwslt2023_s2t_en-de > iwslt2023_s2t_en-de.tar

```

For a submission, for instance on `en-de`, following three items are needed:

- docker image `iwslt2023_s2t_en-de.tar`
- results on `tst-COMMON`, `${OUTPUT_DIR}-De`, saved as `results`
- A readme file.

The example submissions can be found here: [en-de](), [en-zh](), [en-ja]()

To validate the docker image, you can start a docker container:

```bash
# Start the docker
docker run -p 2023:2023 --gpus all --shm-size 32G iwslt2023_s2t_en-de:latest

# Run evaluation
simuleval --remote-eval \
    --source data/wavs.txt --target data/refs.txt \
    --source-type speech --target-type text \
    --source-segment-size 160 \
    --end-index 10 \
    --output ${docker-output}
```

The `${docker-output}` should have the same output in [Inference & Evaluation](#inference--evaluation) section.
