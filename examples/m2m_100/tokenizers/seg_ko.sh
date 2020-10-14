#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
SCRIPT=`realpath $0`
MECAB=`dirname $SCRIPT`/thirdparty/mecab-0.996-ko-0.9.2

export PATH=$PATH:"$MECAB/bin":"$MECAB/lib"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$MECAB/lib"

cat - | mecab -O wakati
