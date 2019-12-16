#!/bin/bash

set -e
set -u


### Path to Moses scripts
moses="/raid/bin/mosesdecoder/scripts/"

### Path to truecasing models
models="/raid/data/daga01/TCmodel/model_1_Cr/modelTC.EpWP.de"

### Path to zipped corpus
#zipped="/raid/data/daga01/data/mt_train/corpus"
#pushd .
#cd "${zipped}"
#tar -xvjf "${zipped}/en-de.wmt19.tar.bz2"
#popd

echo "Start processing"

### Path to the unzipped corpus
top_corp="/raid/data/daga01/data/mt_train/corpus/original"

files=(commoncrawl.de  en-de.bicleaner07.lanClean.tmp33.l50.de  EP.de  nc.de  rapid.Umlauts.de)

### Language codes
l1_code="de"
l2_code="en"

for i in "${files[@]}"
do
    file=${top_corp}/$i
    python3 run_all.py $moses $models $file "$l1_code" "$l2_code" --num-threads 4 -ld -capl -tu &
done

