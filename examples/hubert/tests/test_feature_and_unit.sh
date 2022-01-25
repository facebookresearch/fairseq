#!/bin/bash

set -e

sizes="base large xlarge"

declare -A ckpt_urls
ckpt_urls[base]="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
ckpt_urls[large]="https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt"
ckpt_urls[xlarge]="https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt"

declare -A km_layers
km_layers[base]=9
km_layers[large]=20
km_layers[xlarge]=30

declare -A km_urls
km_urls[base]="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin"

declare -A km_nunits
km_nunits[base]=500

test_dir=./examples/hubert/tests
split=sample


check_feature () {
  echo "checking features..."

  size=$1
  ckpt_url=$2
  km_layer=$3
  ckpt_path="$test_dir/$(basename "$ckpt_url")"

  if [ ! -f "$ckpt_path" ]; then
    echo "downloading $ckpt_url to $ckpt_path"
    wget "$ckpt_url" -O "$ckpt_path"
  fi

  python ./examples/hubert/simple_kmeans/dump_hubert_feature.py \
    "${test_dir}" "${split}" "${ckpt_path}" "${km_layer}" 1 0 "${test_dir}"

  if diff -q "${test_dir}/${split}.${size}.L${km_layer}.npy" "${test_dir}/${split}_0_1.npy" &>/dev/null; then
    echo "...passed npy check"
  else
    echo "...failed npy check"
  fi

  if diff -q "${test_dir}/${split}.${size}.L${km_layer}.len" "${test_dir}/${split}_0_1.len" &>/dev/null; then
    echo "...passed len check"
  else
    echo "...failed len check"
  fi
}


check_unit () {
  echo "checking units..."

  size=$1
  km_url=$2
  km_layer=$3
  km_nunit=$4
  km_path="$test_dir/$(basename "$km_url")"

  if [ ! -f "$km_path" ]; then
    echo "downloading $km_url to $km_path"
    wget "$km_url" -O "$km_path"
  fi

  python ./examples/hubert/simple_kmeans/dump_km_label.py \
    "${test_dir}" "${split}" "${km_path}" 1 0 "${test_dir}"

  if diff -q "${test_dir}/${split}.${size}.L${km_layer}.km${km_nunit}.km" "${test_dir}/${split}_0_1.km" &>/dev/null; then
    echo "...passed unit check"
  else
    echo "...failed unit check"
  fi
}


for size in $sizes; do
  echo "=== Running unit test for HuBERT $size ==="
  check_feature "$size" "${ckpt_urls[$size]}" "${km_layers[$size]}"

  if [ -n "${km_urls[$size]}" ]; then
    check_unit "$size" "${km_urls[$size]}" "${km_layers[$size]}" "${km_nunits[$size]}"
  fi

  rm -f $test_dir/${split}_0_1.*
done
