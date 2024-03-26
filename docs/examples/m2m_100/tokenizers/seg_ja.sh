#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
SCRIPT=`realpath $0`
KYTEA=`dirname $SCRIPT`/thirdparty/kytea
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KYTEA/lib:/usr/local/lib
export PATH=$PATH:"$KYTEA/bin"

cat - | tr -d "[:blank:]" | kytea -notags
