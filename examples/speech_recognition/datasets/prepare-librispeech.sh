#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Prepare librispeech dataset

base_url=www.openslr.org/resources/12
train_dir=train_960

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <download_dir> <out_dir>"
  echo "e.g.: $0 /tmp/librispeech_raw/ ~/data/librispeech_final"
  exit 1
fi

download_dir=${1%/}
out_dir=${2%/}

fairseq_root=~/fairseq-py/
mkdir -p ${out_dir}
cd ${out_dir} || exit

nbpe=5000
bpemode=unigram

if [ ! -d "$fairseq_root" ]; then
    echo "$0: Please set correct fairseq_root"
    exit 1
fi

echo "Data Download"
for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    url=$base_url/$part.tar.gz
    if ! wget -P $download_dir $url; then
        echo "$0: wget failed for $url"
        exit 1
    fi
    if ! tar -C $download_dir -xvzf $download_dir/$part.tar.gz; then
        echo "$0: error un-tarring archive $download_dir/$part.tar.gz"
        exit 1
    fi
done

echo "Merge all train packs into one"
mkdir -p ${download_dir}/LibriSpeech/${train_dir}/
for part in train-clean-100 train-clean-360 train-other-500; do
    mv ${download_dir}/LibriSpeech/${part}/* $download_dir/LibriSpeech/${train_dir}/
done
echo "Merge train text"
find ${download_dir}/LibriSpeech/${train_dir}/ -name '*.txt' -exec cat {} \; >> ${download_dir}/LibriSpeech/${train_dir}/text

# Use combined dev-clean and dev-other as validation set
find ${download_dir}/LibriSpeech/dev-clean/ ${download_dir}/LibriSpeech/dev-other/ -name '*.txt' -exec cat {} \; >> ${download_dir}/LibriSpeech/valid_text
find ${download_dir}/LibriSpeech/test-clean/ -name '*.txt' -exec cat {} \; >> ${download_dir}/LibriSpeech/test-clean/text
find ${download_dir}/LibriSpeech/test-other/ -name '*.txt' -exec cat {} \; >> ${download_dir}/LibriSpeech/test-other/text


dict=data/lang_char/${train_dir}_${bpemode}${nbpe}_units.txt
encoded=data/lang_char/${train_dir}_${bpemode}${nbpe}_encoded.txt
fairseq_dict=data/lang_char/${train_dir}_${bpemode}${nbpe}_fairseq_dict.txt
bpemodel=data/lang_char/${train_dir}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
echo "Dictionary preparation"
mkdir -p data/lang_char/
echo "<unk> 3" > ${dict}
echo "</s> 2" >> ${dict}
echo "<pad> 1" >> ${dict}
cut -f 2- -d" " ${download_dir}/LibriSpeech/${train_dir}/text > data/lang_char/input.txt
spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --unk_id=3 --eos_id=2 --pad_id=1 --bos_id=-1 --character_coverage=1
spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt > ${encoded}
cat ${encoded} | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+3}' >> ${dict}
cat ${encoded} | tr ' ' '\n' | sort | uniq -c | awk '{print $2 " " $1}' > ${fairseq_dict}
wc -l ${dict}

echo "Prepare train and test jsons"
for part in train_960 test-other test-clean; do
    python ${fairseq_root}/examples/speech_recognition/datasets/asr_prep_json.py --audio-dirs ${download_dir}/LibriSpeech/${part} --labels ${download_dir}/LibriSpeech/${part}/text --spm-model ${bpemodel}.model --audio-format flac --dictionary ${fairseq_dict} --output ${part}.json
done
# fairseq expects to find train.json and valid.json during training
mv train_960.json train.json

echo "Prepare valid json"
python ${fairseq_root}/examples/speech_recognition/datasets/asr_prep_json.py --audio-dirs ${download_dir}/LibriSpeech/dev-clean ${download_dir}/LibriSpeech/dev-other --labels ${download_dir}/LibriSpeech/valid_text --spm-model ${bpemodel}.model --audio-format flac --dictionary ${fairseq_dict} --output valid.json

cp ${fairseq_dict} ./dict.txt
cp ${bpemodel}.model ./spm.model
