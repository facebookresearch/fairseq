# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# ROOT=$(dirname "$0")
MAIN_PATH=$PWD
ROOT=$MAIN_PATH/$(dirname "$0")
TOOL_DIR=${ROOT}/../../../../tools

echo $TOOL_DIR

mkdir -p $TOOL_DIR

INDICNLP=$TOOL_DIR/indic_nlp_library
if [ ! -e $INDICNLP ]; then
    echo "Cloning Indic NLP Library..."
    git -C $TOOL_DIR clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
    pushd $INDICNLP
    git reset --hard 0a5e01f2701e0df5bc1f9905334cd7916d874c16
    popd
else
    echo "Indic is already pulled from github. Skipping."
fi

# need python2 :(
apt-get update
apt-get  -y install python2

