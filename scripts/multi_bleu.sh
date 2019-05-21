#!/bin/bash

if [ $# -ne 2 ]; then
    echo "usage: $0 GENERATE_PY_OUTPUT PROCESS_ATAT_FLAG"
    exit 1
fi

echo -e "\n RUN >> "$0" -- ATAT $2"


GEN=$1
DO_ATAT=$2

SYS=$GEN.sys
REF=$GEN.ref

if [ $(tail -n 1 $GEN | grep BLEU | wc -l) -ne 1 ]; then
    echo "not done generating"
    exit
fi

if [ "$DO_ATAT" == "1" ]; then
    grep ^H $GEN | cut -d - -f 2- | sort -n -k 1 | cut -f 3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
    grep ^T $GEN | cut -d - -f 2- | sort -n -k 1 | cut -f 2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
else
    grep ^H $GEN | cut -d - -f 2- | sort -n -k 1 | cut -f 3- > $SYS
    grep ^T $GEN | cut -d - -f 2- | sort -n -k 1 | cut -f 2- > $REF
fi
perl ./scripts/multi-bleu.perl $REF < $SYS
