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
exp_tag=           # exp tags for multiple exp
expdir=exp         # experiments directory
dumpdir=dump       # feature dump directory
src_audio=         # a directory of all source audios (in wav format)
tgt_audio=         # a directory of all target audios (in wav format)
src_text=          # source text forlder contains Kaldi-style text for every subset
                   # Kaldi-style text (e.g., text.subset.lang)
                   # utt_id word1 word2 ...
tgt_text=          # target text forlder contains Kaldi-style text for every subset
train_set=train    # Training subset
dev_set=dev        # Development subset
test_set=test      # Test subset
dev_size=500       # Developmenet subset size (for faster training, use smaller DEV size)
split="${train_set} ${dev_set} ${test_set}"

# feature related
fs=16000           # for discrete unit-based method, sampling rate recommends to be 16000
                   # for spectrogram-based method, sampling rate recommends to be 22050

# model related
model_architect=s2ut    # model architecture name
n_frame_per_step=5      # num of stacked frames, use 0 for reduced discrete unit sequence
use_mtl=true            # whether to use multi-task learning (src, decoder_ctc, decoder)

# training related
optimizer=adam          # training optimizer
lr=0.0005               # learning rate
dropout=0.1             # dropout rate
warmup_updates=10000    # warmup steps
max_update=400000       # maximum update
max_tokens=80000        # maximum tokens per iteration
update_freq=16          # accum grad (simulate multi-gpu)
seed=1                  # random seed

# discrete unit extraction (only for s2ut model)
hubert_checkpoint=hubert.pt   # hubert checkpoint
                              # default use hubert_base_960
hubert_layer=6                # hubert features for feature extraction
km_model=km.bin               # k-means model for discrete unit clustering
cluster_km=100                # number of k-means clusters

# code-based vocoder
vocoder_checkpoint=code-hifigan.pt    # code-based vocoder
vocoder_config=vocoder_config.json    # config for code-based vocoder

# evaluation
recognizer=recognizer        # name for recognizer

# ========== Data Preparation =============

pretrained_dir=${expdir}/pre-trained_model   # directory for pre-trained models
model_dir=${expdir}/${model_architect}_${exp_tag}       # model save directory


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Prepare pre-trained models"
    # Skip the stage if use another pre-trained models

    # Hubert model
    log "Download Hubert-base trained on LibriSpeech960"
    wget -O ${pretrained_dir}/${hubert_checkpoint} https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt

    # Hubert kmeans model
    log "Download K-means cluster for discrete unit generation"
    wget -O ${pretrained_dir}/${km_model} https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin

    # code-based vocoder
    log "Download code-based Hifi-GAN vocoder"
    wget -O ${pretrained_dir}/${vocoder_checkpoint} https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000
    wget -O ${pretrained_dir}/${vocoder_config} https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/config.json

    # No need for current interface
    # # wav2vec2 ASR (for evaluation)
    # log "Download wav2vec2 ASR"
    # wget -O ${pretrained_dir}/${recognizer} https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Speech synthesis for target speech"

    # Resulting Format:
    # ${tgt_audio}
    #   - train
    #     - ${sample_id}.wav
    #     - ...
    #   - dev
    #   - test

    # speech filter by ASR error?
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ "${model_architect}" == "s2ut" ] ; then
    log "Stage 2: Extract discrete unit (only for s2ut-based model)"

    # downsample and create manifest for hubert unit extraction
    python examples/speech_to_speech/recipe/scripts/downsample_create_manifest.py \
        --src ${tgt_audio} \
        --tgt ${tgt_audio}_downsampled \
        --subset "${train_set}" \
        --subset "${dev_set}" \
        --subset "${test_set}" \
        --dev_size ${dev_size} \
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
    ptyhon examples/speech_to_speech/recipe/scripts/prepare_mtl.py \
        --src_text ${src_text} \
        --tgt_text ${tgt_text} \
        --subset "${train_set}" \
        --subset "${dev_set}" \
        --subset "${test_set}" \
        --dumpdir ${dumpdir}
    
    # copy identical multi-task
    cp -r ${dumpdir}/target_letter ${dumpdir}/decoder_target_ctc

    # add softlink for config
    if [[ "${model_architect}" == "s2ut" ]]; then
        ln -sf examples/speech_to_speech/recipe/conf/config_multitask_ut.yaml ${dumpdir}/${model_architect}
    elif [[ "${model_architect}" == "s2spect" ]]; then
        ln -sf examples/speech_to_speech/recipe/conf/config_multitask_spect.yaml ${dumpdir}/${model_architect}
    else
        log "Not supported model architecture ${model_architect}"
    fi
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
          --results-path ${result_path}/unit

        grep "^D\-" ${RESULTS_PATH}/generate-${test_set}.txt | \
          sed 's/^D-//ig' | sort -nk1 | cut -f3 \
          > ${RESULTS_PATH}/generate-${test_set}.unit

        python examples/speech_to_speech/generate_waveform_from_code.py \
          --in-code-file ${result_path}/generate-${test_set}.unit \
          --vocoder ${pretrained_dir}/${vocoder_checkpoint} --vocoder-cfg ${pretrained_dir}/${vocoder_config} \
          --results-path ${result_path}/wav --dur-prediction

    elif [[ "${model_architect}" == "s2spect" ]]; then
        # assume using a default Griffin-Lim vocoder

        python examples/speech_synthesis/generate_waveform.py ${dumpdir}/${model_architect} \
          --config-yaml config.yaml --multitask-config-yaml config_multitask.yaml \
          --task speech_to_speech --n-frames-per-step 5 \
          --path ${model_dir}}/checkpoint_best.pt  --gen-subset ${test_set}} \
          --max-tokens 50000 \
          --results-path ${result_path}/wav --dump-waveforms --output-sample-rate 16000

    else
        log "Not supported model architecture ${model_architect}"
    fi
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    log "Stage 7: Model Evaluation Prepare (ASR recognition)"
    # recognizer
    python examples/speech_to_speech/recipe/scripts/wav2vec2_en_recognizer.py \
        --audios ${result_path} \
        --extension ".wav" \
        --recognized_output ${result_path}/score/recognized.txt

    # text cleaner (we use espnet_tts_frontend for tacotron_cleaner)
    if [[ "${model_architect}" == "s2ut" ]]; then
        python examples/speech_to_speech/recipe/scripts/text_clean.py \
            --recognized_output ${result_path}/score/recognized.txt \
            --sort
    else
        python examples/speech_to_speech/recipe/scripts/text_clean.py \
            --recognized_output ${result_path}/score/recognized.txt
    fi

    python examples/speech_to_speech/recipe/scripts/text_clean.py
        --recognized_output ${src_text}/text.${test_set} \
        --output_path ${result_path}/score/ref.txt.cleaned
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    log "Stage 8: Scoring"
    detokenizer.perl -l en -q < "${result_path}/score/ref.txt.cleaned" \
        > "${result_path}/score/ref.txt.cleaned.detok"
    detokenizer.perl -l en -q < "${result_path}/score/recognized.txt.cleaned" \
        > "${result_path}/score/recognized.txt.cleaned"

    sacrebleu -lc "${result_path}/score/ref.txt.cleaned.detok" \
                -i "${result_path}/score/recognized.txt.cleaned" \
                -m bleu chrf ter \
                >> ${result_path}/score/result.lc.txt
fi
