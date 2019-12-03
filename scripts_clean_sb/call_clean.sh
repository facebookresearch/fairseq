#!/bin/bash

set -e
set -u

moses="/raid/bin/mosesdecoder/scripts/"
models="/raid/data/daga01/TCmodel/model_1_Cr/modelTC.EpWP.de"

#zipped="/raid/data/daga01/data/mt_train/corpus"
#pushd .
#cd "${zipped}"
#tar -xvjf "${zipped}/en-de.wmt19.tar.bz2"
#popd

echo "Start processing"

top_corp="/raid/data/daga01/data/mt_train/corpus/original"

#top_corp="/raid/data/daga01/data/mt_train/cleanedDir"
files=(commoncrawl.de  en-de.bicleaner07.lanClean.tmp33.l50.de  EP.de  nc.de  rapid.Umlauts.de)

#top_corp="/raid/data/daga01/data/mt_train/test_clean"
#files=(commoncrawl.de)

l1_code="de"
l2_code="en"

for i in "${files[@]}"
do
    file=${top_corp}/$i
    python3 run_all.py $moses $models $file "$l1_code" "$l2_code" --num-threads 4 -ld -capl -tu
done

