#!/usr/bin/env bash

out_root=/tmp
out_name=train_${RANDOM}
num_nonsil_states=1

valid="dev_other"
train="train"
mono_size="-1"  # 2000
tri1_size="-1"  # 5000
tri2b_size="-1"  # 10000
tri3b_size="-1"  # 10000

# Acoustic model parameters
numLeavesTri1=2000
numGaussTri1=10000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000

stage=1
max_stage=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

data=$1
lang=$2
lang_test=$3

exp_root=$out_root/$out_name

# you might not want to do this for interactive shells.
set -e


if [ $stage -le 1 ] && [ $max_stage -ge 1 ]; then
  # train a monophone system
  if [ ! $mono_size -eq -1 ]; then
    utils/subset_data_dir.sh $data/$train $mono_size $data/${train}_${mono_size}
    mono_train=${train}_${mono_size}
  else
    mono_train=${train}
  fi

  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    --initial-beam 40 --regular-beam 60 --retry-beam 120 \
    $data/$mono_train $lang $exp_root/mono

  utils/mkgraph.sh $lang_test $exp_root/mono $exp_root/mono/graph
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    $exp_root/mono/graph $data/$valid $exp_root/mono/decode_$valid &
fi


if [ $stage -le 2 ] && [ $max_stage -ge 2 ]; then
  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  if [ ! $tri1_size -eq -1 ]; then
    utils/subset_data_dir.sh $data/$train $tri1_size $data/${train}_${tri1_size}
    tri1_train=${train}_${tri1_size}
  else
    tri1_train=${train}
  fi

  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
    $data/$tri1_train $lang \
    $exp_root/mono $exp_root/mono_ali_${tri1_train}

  steps_gan/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
      --num_nonsil_states $num_nonsil_states $numLeavesTri1 $numGaussTri1 \
      $data/$tri1_train $lang \
      $exp_root/mono_ali_${tri1_train} $exp_root/tri1

  utils/mkgraph.sh $lang_test $exp_root/tri1 $exp_root/tri1/graph
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    $exp_root/tri1/graph $data/$valid $exp_root/tri1/decode_$valid &
fi

if [ $stage -le 3 ] && [ $max_stage -ge 3 ]; then
  # train an LDA+MLLT system.
  if [ ! $tri2b_size -eq -1 ]; then
    utils/subset_data_dir.sh $data/$train $tri2b_size $data/${train}_${tri2b_size}
    tri2b_train=${train}_${tri2b_size}
  else
    tri2b_train=${train}
  fi

  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
    $data/$tri2b_train $lang \
    $exp_root/tri1 $exp_root/tri1_ali_${tri2b_train}

  steps_gan/train_lda_mllt.sh --cmd "$train_cmd" \
      --num_nonsil_states $num_nonsil_states \
      --splice-opts "--left-context=3 --right-context=3" $numLeavesMLLT $numGaussMLLT \
      $data/$tri2b_train $lang \
      $exp_root/tri1_ali_${tri2b_train} $exp_root/tri2b

  utils/mkgraph.sh $lang_test $exp_root/tri2b $exp_root/tri2b/graph
  steps/decode.sh --nj 20 --cmd "$decode_cmd" \
    $exp_root/tri2b/graph $data/$valid $exp_root/tri2b/decode_$valid &
fi


if [ $stage -le 4 ] && [ $max_stage -ge 4 ]; then
  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  if [ ! $tri3b_size -eq -1 ]; then
    utils/subset_data_dir.sh $data/$train $tri3b_size $data/${train}_${tri3b_size}
    tri3b_train=${train}_${tri3b_size}
  else
    tri3b_train=${train}
  fi

  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
    $data/$tri3b_train $lang \
    $exp_root/tri2b $exp_root/tri2b_ali_${tri2b_train}

  steps_gan/train_sat.sh --cmd "$train_cmd" \
    --num_nonsil_states $num_nonsil_states $numLeavesSAT $numGaussSAT \
    $data/$tri3b_train $lang \
    $exp_root/tri2b_ali_${tri2b_train} $exp_root/tri3b

  utils/mkgraph.sh $lang_test $exp_root/tri3b $exp_root/tri3b/graph
  steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
    $exp_root/tri3b/graph $data/$valid $exp_root/tri3b/decode_$valid &
fi

wait
