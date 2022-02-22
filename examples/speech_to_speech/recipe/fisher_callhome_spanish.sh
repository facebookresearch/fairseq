#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


SECONDS=0

# General configuration
stage=1
stop_stage=100
expdir=exp
dumpdir=dump
src_audio=
tgt_audio=
src_text=
tgt_text=
train_set=train
dev_set=dev
test_set=test 
split="${train_set} ${dev_set} ${test_set}"

# feature related
fs=16000

# model related
model_architect=s2ut
n_frame_per_step=5
use_mtl=true

# training related
optimizer=adam
lr=0.0005
dropout=0.1
warmup_updates=10000
max_update=400000
max_tokens=80000
update_freq=16
seed=1

# discrete unit extraction (only for s2ut model)
hubert_checkpoint=hubert.pt
hubert_layer=6
km_model=km.bin
cluster_km=100

# code-based vocoder
vocoder_checkpoint=code-hifigan.pt
vocoder_config=vocoder_config.json

# ========== Data Preparation =============

pretrained_dir=${expdir}/pre-trained_model
model_dir=${expdir}/${model_architect}


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Prepare pre-trained models"
    # Skip the stage if use another pre-trained models

    # Hubert model
    wget -O ${pretrained_dir}/${hubert_checkpoint} https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt

    # Hubert kmeans model
    wget -O ${pretrained_dir}/${km_model} https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin

    # code-based vocoder
    wget -O ${pretrained_dir}/${vocoder_checkpoint} https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000
    wget -O ${pretrained_dir}/${vocoder_config} https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Speech synthesis for target speech"
    # TODO

    # Resulting Format:
    # ${tgt_audio}
    #   - train
    #     - ${sample_id}.wav
    #     - ...
    #   - dev
    #   - test
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${model_architect}" == "s2ut" ] ; then
    log "Stage 2: Extract discrete unit (only for s2ut-based model)"

    # downsample and create manifest for hubert unit extraction
    # TODO (jiatong)
    python downsample_create_manifest.py \
        --src ${tgt_audio} \
        --tgt ${tgt_audio}_downsampled \
        --split "${split}" \
        --fs ${fs}

    for set in split; do
        python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
            --feature_type hubert \
            --kmeans_model_path ${pretrained_dir}/${km_model} \
            --acoustic_model_path ${pretrained_dir}/${hubert_checkpoint} \
            --layer ${hubert_layer} \
            --manifest_path ${tgt_audio}_downsampled/manifest_${set}.tsv \
            --out_quantized_file_path ${tgt_audio}_downsampled/${set}.txt \
            --extension ".wav"
    done
fi

# change variable accordingly
if [[ "${model_architect}" == "s2ut" ]]; then
    tgt_audio=${tgt_audio}_downsampled
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Formating Data"
    if [[ "${model_architect}" == "s2ut" ]]; then
        python examples/speech_to_speech/preprocessing/prep_s2ut_data.py \
            --source-dir ${src_audio} --target-dir ${tgt_audio} --data-split ${split} \
            --output-root ${dumpdir}/${model_architect} --reduce-unit \
            --vocoder-checkpoint ${pretrained_dir}/${vocoder_checkpoint} \
            --vocoder-cfg ${pretrained_dir}/${vocoder_config}
    elif [[ "${model_architect}" == "s2spect" ]]; then
    python examples/speech_to_speech/preprocessing/prep_s2spect_data.py \
            --source-dir ${src_audio} --target-dir ${tgt_audio} --data-split ${split} \
            --output-root ${dumpdir}/${model_architect}
    else
        log "Not supported model architecture ${model_architect}"
    fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && [ "${use_mtl}" == "true" ]; then
    log "Stage 4: Generating multi-task task"
    ptyhon generate_mtl.py \
        --src_text ${src_text} \
        --tgt_text ${tgt_text} \
        --split ${split} \
        --dumpdir ${dumpdir}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Model Training"
    if [[ "${model_architect}" == "s2ut" ]]; then
        fairseq-train ${dumpdir}/${model_architect} \
          --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
          --task speech_to_speech --target-is-code --target-code-size ${cluster_km} --vocoder code_hifigan  \
          --criterion speech_to_unit --label-smoothing 0.2 \
          --arch s2ut_transformer_fisher --share-decoder-input-output-embed \
          --dropout ${dropout} --attention-dropout ${dropout} --relu-dropout ${dropout} \
          --train-subset ${train_set} --valid-subset ${dev} \
          --save-dir ${model_dir} \
          --lr ${lr} --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates ${warmu_updates} \
          --optimizer ${optimizer}  --adam-betas "(0.9,0.98)" --clip-norm 10.0 \
          --max-update ${max_update} --max-tokens ${max_tokens} --max-target-positions 3000 --update-freq ${update_freq} \
          --seed 1 --fp16 --num-workers 8

    elif [[ "${model_architect}" == "s2spect" ]]; then
        fairseq-train ${dumpdir}/${model_architect} \
          --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
          --task speech_to_speech --n-frames-per-step 5 \
          --criterion speech_to_spectrogram \
          --arch s2spect_transformer_fisher --decoder-normalize-before \
          --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
          --train-subset train --valid-subset dev \
          --save-dir ${model_dir} \
          --eval-inference --best-checkpoint-metric mcd_loss \
          --lr ${lr} --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates ${warmu_updates} \
          --optimizer ${optimizer} --adam-betas "(0.9,0.98)" --clip-norm 10.0 --weight-decay 1e-6 \
          --max-update  ${max_update} --max-tokens ${max_tokens} --max-tokens-valid 30000  --required-batch-size-multiple 1 \
          --max-target-positions 3000 --update-freq ${update_freq} \
          --seed 1 --fp16 --num-workers 8

    else
        log "Not supported model architecture ${model_architect}"
    fi
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Model Inference"
    result_path=${expdir}/${model_architect}_decode_${test_set}

    if [[ "${model_architect}" == "s2ut" ]]; then
        fairseq-generate ${dumpdir}/${model_architect} \
          --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
          --task speech_to_speech --target-is-code --target-code-size 100 --vocoder code_hifigan \
          --path ${model_dir}/checkpoint_best.pt  --gen-subset ${test_set} \
          --max-tokens 50000 \
          --beam 10 --max-len-a 1 \
          --results-path ${result_path}

        grep "^D\-" ${RESULTS_PATH}/generate-${test_set}.txt | \
          sed 's/^D-//ig' | sort -nk1 | cut -f3 \
          > ${RESULTS_PATH}/generate-${test_set}.unit

        python examples/speech_to_speech/generate_waveform_from_code.py \
          --in-code-file ${result_path}/generate-${test_set}.unit \
          --vocoder ${pretrained_dir}/${vocoder_checkpoint} --vocoder-cfg ${pretrained_dir}/${vocoder_config} \
          --results-path ${result_path} --dur-prediction

    elif [[ "${model_architect}" == "s2spect" ]]; then
        # assume using a default Griffin-Lim vocoder

        python examples/speech_synthesis/generate_waveform.py ${dumpdir}/${model_architect} \
          --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
          --task speech_to_speech --n-frames-per-step 5 \
          --path ${model_dir}}/checkpoint_best.pt  --gen-subset ${test_set}} \
          --max-tokens 50000 \
          --results-path ${result_path} --dump-waveforms --output-sample-rate 16000

    else
        log "Not supported model architecture ${model_architect}"
    fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Model Evaluation and Scoring"
    # TODO
fi


