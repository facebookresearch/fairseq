#!/bin/bash

set -e

sizes="large xlarge"

declare -A ckpt_urls
ckpt_urls[large]="https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k_finetune_ls960.pt"
ckpt_urls[xlarge]="https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k_finetune_ls960.pt"

test_dir=$(pwd)/examples/hubert/tests
split=sample

echo -e "${test_dir}\n6313-76958-0021.flac\t190800" > "${test_dir}/${split}.tsv"
echo -e "K E E P | A | G O I N G | A N D | I F | Y O U ' R E | L U C K Y | Y O U ' L L | R U N | P L U M B | I N T O | T H E M | W A S | T H E | J E E R I N G | A N S W E R | A S | T H E | S L E E P Y | C O W M E N | S P U R R E D | T H E I R | P O N I E S | O N | T O W A R D | C A M P | M U T T E R I N G | T H E I R | D I S A P P R O V A L | O F | T A K I N G | A L O N G | A | B U N C H | O F | B O Y S | O N | A | C A T T L E | D R I V E |" > "${test_dir}/${split}.ltr"

check_asr () {
  echo "checking asr outputs..."

  size=$1
  ckpt_url=$2
  ckpt_path="$test_dir/$(basename "$ckpt_url")"

  if [ ! -f "$ckpt_path" ]; then
    echo "downloading $ckpt_url to $ckpt_path"
    wget "$ckpt_url" -O "$ckpt_path"
  fi

  python examples/speech_recognition/new/infer.py \
    --config-dir examples/hubert/config/decode --config-name infer_viterbi \
    common_eval.path="${ckpt_path}" task.data="${test_dir}" task.normalize=true \
    decoding.results_path="${test_dir}/pred" \
    common_eval.results_path="${test_dir}/pred" \
    common_eval.quiet=false dataset.gen_subset="${split}"

  if diff -q "${test_dir}/pred/hypo.word" "${test_dir}/${split}.${size}.hypo.word" &>/dev/null; then
    echo "...passed word check"
  else
    echo "...failed word check"
  fi
  rm -rf "${test_dir}/pred"
}

for size in $sizes; do
  check_asr "$size" "${ckpt_urls[$size]}"
done
