#!/bin/bash

# Fine-tuning of BART model on CNN-DM
set -eu
fairseq_root="/fsx/akshats/github/fairseq_public/fairseq"
data_path_list=(
  "/fsx/akshats/data/stop_fairseq_seq2seq/full_dataset-bin"
)
name_list=(
  "bart_1_gpu_beam_size_1"
)
dataset=stop
task=bart_metrics

exp_root_dir="/fsx/${USER}/checkpoints/nlu/${dataset}/${task}"

num_gpus=1

for i in ${!data_path_list[@]}; do
  data_path=${data_path_list[$i]}
  name=${name_list[$i]}
  run_dir="${exp_root_dir}/${name}"
  mkdir -p $run_dir

  echo "#### BART FINE-TUNING: $name"
  echo "output dir: $run_dir"
  CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=WARN fairseq-train $data_path \
    --arch bart_base \ 
    --task translation 
    # --criterion label_smoothed_cross_entropy \
    # --source-lang utterance \
    # --target-lang parse_decoupled_bart \
    # --truncate-source \ 
    # --label-smoothing 0.0 \ 
    # --valid-subset test_music,test_messaging \
    # --train-subset eval_no \
    # --max-tokens 2048 \
    # --update-freq 1 \
    # --max-update 0 \
    # --max-epoch 1 \
    # --batch-size 32 \
    # --required-batch-size-multiple 1 \
    # --dropout 0.2 \
    # --attention-dropout 0.1 \
    # --relu-dropout 0.0 \
    # --weight-decay 0 \
    # --optimizer adam \
    # --adam-betas '"'"'(0.9, 0.98)'"'"' \
    # --adam-eps 1e-10 \
    # --clip-norm 0.0 \
    # --lr-scheduler inverse_sqrt \
    # --lr 0 \ 
    # --warmup-updates 5000 \
    # --eval-bleu-args '"'"'{"beam": 1, "max_len_b": 140}'"'"' \
    # --eval-bleu-detok moses \
    # --eval-bleu-remove-bpe \
    # --num-workers 4 \
    # --no-save \
    # --no-epoch-checkpoints \
    # --reset-meters \
    # --reset-dataloader \
    # --reset-optimizer \
    # --share-all-embeddings \
    # --layernorm-embedding \
    # --share-decoder-input-output-embed \
    # --skip-invalid-size-inputs-valid-test \
    # --log-format json \
    # --log-interval 10
done