# Pay Better Attention to Attention: Head Selection in Multilingual and Multi-Domain Sequence Modeling (Gong et al., 2021)

[https://arxiv.org/pdf/2106.10840.pdf](https://arxiv.org/pdf/2106.10840.pdf)

## Introduction

We present attention head selection strategies in multilingual and multi-domain sequence modeling including text translation, speech recognition and speech translation tasks.

Below is an example of training multilingual/multi-domain speech recognition models.

## Data Preparation
Prepare mTEDx data as in [mTEDx example](https://github.com/fairinternal/fairseq-py/blob/0d9c5851e6fac40f9e366b3633ccd615c2901788/examples/speech_to_text/docs/mtedx_example.md) and CoVoST data as in [CoVoST example](https://github.com/fairinternal/fairseq-py/blob/0d9c5851e6fac40f9e366b3633ccd615c2901788/examples/speech_to_text/docs/covost_example.md). Similarly prepare EuroParl data.


## Training a multilingual ASR model with attention head selection

```bash
data_dir=<path to mtedx data>
train_subset="train_ar_ar_tedx,train_de_de_tedx,train_el_el_tedx,train_es_es_tedx,train_fr_fr_tedx,train_it_it_tedx,train_pt_pt_tedx,train_ru_ru_tedx"
valid_subset="valid_ar_ar_tedx,valid_de_de_tedx,valid_el_el_tedx,valid_es_es_tedx,valid_fr_fr_tedx,valid_it_it_tedx,valid_pt_pt_tedx,valid_ru_ru_tedx"
strateg=<subset or group>

fairseq-train ${data_dir} \
    --user-dir examples/attention_head_selection/src \
    --train-subset "${train_subset}" \
    --valid-subset "${valid_subset}" \
    --config-yaml 'config_asr.yaml' \
    --arch 'head_selection_s2t_transformer_s' \
    --task 'speech_to_text_head_selection' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler 'inverse_sqrt' --stop-min-lr -1.0 --warmup-updates 10000 \
    --lr 5e-4 \
    --clip-norm 10.0 \
    --seed 1 \
    --max-epoch 400 \
    --max-tokens 32000 \
    --ignore-prefix-size 1 \
    --dropout 0.3 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --skip-invalid-size-inputs-valid-test \
    --encoder-attn-head-select \
    --total-encoder-attention-heads 8 \
    --decoder-self-attn-head-select \
    --total-decoder-attention-heads 8 \
    --attn-head-select-strategy ${strategy} \
    --task-type lang \
```

## Training a multi-domain ASR model with attention head selection

```bash
data_dir=<path to multi-domain data>
train_subset="train_es_es_tedx,train_fr_fr_tedx,train_pt_pt_tedx,train_it_it_tedx,train_ru_ru_tedx,train_el_el_tedx,train_ar_ar_tedx,train_de_de_tedx,train_ar_ar_cv,train_de_de_cv,train_es_es_cv,train_fr_fr_cv,train_it_it_cv,train_pt_pt_cv,train_ru_ru_cv,train_de_de_ep,train_es_es_ep,train_fr_fr_ep,train_it_it_ep,train_pt_pt_ep"
valid_subset="dev_es_es_tedx,dev_fr_fr_tedx,dev_pt_pt_tedx,dev_it_it_tedx,dev_ru_ru_tedx,dev_el_el_tedx,dev_ar_ar_tedx,dev_de_de_tedx,dev_ar_ar_cv,dev_de_de_cv,dev_es_es_cv,dev_fr_fr_cv,dev_it_it_cv,dev_pt_pt_cv,dev_ru_ru_cv,dev_de_de_ep,dev_es_es_ep,dev_fr_fr_ep,dev_it_it_ep,dev_pt_pt_ep"
strateg=<subset or group>

fairseq-train ${data_dir} \
    --user-dir examples/attention_head_selection/src \
    --train-subset "${train_subset}" \
    --valid-subset "${valid_subset}" \
    --config-yaml 'config_asr.yaml' \
    --arch head_selection_s2t_transformer_s \
    --task speech_to_text_head_selection \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler 'inverse_sqrt' --stop-min-lr -1.0 --warmup-updates 10000 \
    --lr 5e-4 \
    --clip-norm 10.0 \
    --seed 1 \
    --max-epoch 400 \
    --max-tokens 32000 \
    --ignore-prefix-size 1 \
    --dropout 0.3 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --skip-invalid-size-inputs-valid-test \
    --encoder-attn-head-select \
    --total-encoder-attention-heads 8 \
    --decoder-self-attn-head-select \
    --total-decoder-attention-heads 8 \
    --attn-head-select-strategy ${strategy} \
    --task-type domain
```

## Inference in multilingual setting

```bash
MODEL_DIR=<checkpoint directory>
data_dir=<path to mtedx data>
gen_subset=<data to test, e.g., test_ar_ar_tedx>
train_subset="train_ar_ar_tedx,train_de_de_tedx,train_el_el_tedx,train_es_es_tedx,train_fr_fr_tedx,train_it_it_tedx,train_pt_pt_tedx,train_ru_ru_tedx"
last_n=10
CHECKPOINT_FILENAME="avg_last_${last_n}_checkpoint.pt"
CHECKPOINT="_avg"
RESULTS="${MODEL_DIR}/ckpt${CHECKPOINT}"
if [ ! -d $RESULTS ]; then
    mkdir -p $RESULTS
fi;

python scripts/average_checkpoints.py \
  --inputs ${MODEL_DIR} --num-epoch-checkpoints ${last_n} \
  --output "${MODEL_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${data_dir} \
    --user-dir examples/attention_head_selection/src \
    --arch 'head_selection_s2t_transformer_s' \
    --task 'speech_to_text_head_selection' \
    --train-subset ${train_subset} \
    --gen-subset ${gen_subset} \
    --path "${MODEL_DIR}/${CHECKPOINT_FILENAME}" \
    --config-yaml 'config_asr.yaml' \
    --prefix-size 1 \
    --max-tokens 40000 --beam 5 \
    --skip-invalid-size-inputs-valid-test \
    --results-path ${RESULTS} \
    --scoring wer --wer-tokenizer 13a \
    --wer-lowercase --wer-remove-punct --remove-bpe
```

## Inference in multi-domain setting

```bash
MODEL_DIR=<checkpoint directory>
data_dir=<path to multi-domain data>
gen_subset=<data to test, e.g., test_pt_pt_cv>
train_subset="train_es_es_tedx,train_fr_fr_tedx,train_pt_pt_tedx,train_it_it_tedx,train_ru_ru_tedx,train_el_el_tedx,train_ar_ar_tedx,train_de_de_tedx,train_ar_ar_cv,train_de_de_cv,train_es_es_cv,train_fr_fr_cv,train_it_it_cv,train_pt_pt_cv,train_ru_ru_cv,train_de_de_ep,train_es_es_ep,train_fr_fr_ep,train_it_it_ep,train_pt_pt_ep"
last_n=10
CHECKPOINT_FILENAME="avg_last_${last_n}_checkpoint.pt"
CHECKPOINT="_avg"
RESULTS="${MODEL_DIR}/ckpt${CHECKPOINT}"
if [ ! -d $RESULTS ]; then
    mkdir -p $RESULTS
fi;

python scripts/average_checkpoints.py \
  --inputs ${MODEL_DIR} --num-epoch-checkpoints ${last_n} \
  --output "${MODEL_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${data_dir} \
    --user-dir examples/attention_head_selection/src \
    --arch 'head_selection_s2t_transformer_s' \
    --task 'speech_to_text_head_selection' \
    --train-subset ${train_subset} \
    --gen-subset ${gen_subset} \
    --path "${MODEL_DIR}/${CHECKPOINT_FILENAME}" \
    --config-yaml 'config_asr.yaml' \
    --prefix-size 1 \
    --max-tokens 40000 --beam 5 \
    --skip-invalid-size-inputs-valid-test \
    --results-path ${RESULTS} \
    --scoring wer --wer-tokenizer 13a \
    --wer-lowercase --wer-remove-punct --remove-bpe
```

## Citation
```bibtex
@article{gong2021attn,
    title={Pay Better Attention to Attention: Head Selection in Multilingual and Multi-Domain Sequence Modeling},
    author={Hongyu Gong and
            Yun Tang and
            Juan Miguel Pino and
            Xian Li},
    journal={arXiv preprint arXiv:2106.10840},
    year={2021}
}
'''
