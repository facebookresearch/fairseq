#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if [ -z $WORKDIR_ROOT ] ;
then
        echo "please specify your working directory root in environment variable WORKDIR_ROOT. Exitting..."
        exit
fi

if [ -z $SPM_PATH ] ;
then
    echo "Please install sentence piecence from https://github.com/google/sentencepiece and set SPM_PATH pointing to the installed spm_encode.py. Exitting..."
    exit
fi

ML50=${WORKDIR_ROOT}/ML50

mkdir -p $ML50/dedup
mkdir -p $ML50/cleaned_dedup

python ./dedup_all.py --from-folder $ML50/raw --to-folder $ML50/dedup
python ./remove_valid_test_in_train.py --from-folder $ML50/dedup --to-folder $ML50/clean
python ./binarize.py --raw-folder $ML50/clean