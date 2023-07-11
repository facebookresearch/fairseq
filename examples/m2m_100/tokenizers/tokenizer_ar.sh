#!/usr/bin/env sh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Please follow the instructions here http://alt.qcri.org/tools/arabic-normalizer/
# to install tools needed for Arabic

echo "Please install Arabic tools: http://alt.qcri.org/tools/arabic-normalizer/"
echo "Then update environment variables in tokenizer_ar.sh"
exit 1

SVMTOOL=...
GOMOSESGO=...
QCRI_ARABIC_NORMALIZER=...

export PERL5LIB="$SVMTOOL/lib":"$GOMOSESGO/bin/MADA-3.2":$PERL5LIB


tempfile=$(mktemp)
cat - > $tempfile

cd $QCRI_ARABIC_NORMALIZER

bash qcri_normalizer_mada3.2_aramorph1.2.1.sh $tempfile
cat $tempfile.mada_norm-aramorph.europarl_tok
