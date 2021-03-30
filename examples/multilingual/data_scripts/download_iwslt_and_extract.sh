#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#echo 'Cloning Moses github repository (for tokenization scripts)...'
#git clone https://github.com/moses-smt/mosesdecoder.git

if [ -z $WORKDIR_ROOT ] ;
then
        echo "please specify your working directory root in environment variable WORKDIR_ROOT. Exitting..."
        exit
fi

 

data_root=${WORKDIR_ROOT}/iwsltv2
DESTDIR=${WORKDIR_ROOT}/ML50/raw


langs="ar_AR it_IT nl_XX ko_KR vi_VN"
echo "data_root: $data_root"

download_path=${data_root}/downloads
raw=${DESTDIR}
tmp=${data_root}/tmp
orig=${data_root}/orig
 
mkdir -p $download_path $orig $raw $tmp
#######################
download_iwslt(){
    iwslt_key=$1
    src=$2
    tgt=$3
    save_prefix=$4
    pushd ${download_path}
    if [[ ! -f ${save_prefix}$src-$tgt.tgz ]]; then
        wget https://wit3.fbk.eu/archive/${iwslt_key}/texts/$src/$tgt/$src-$tgt.tgz -O ${save_prefix}$src-$tgt.tgz
        [ $? -eq 0 ] && return 0
    fi         
    popd
}

extract_iwslt(){
    src=$1
    tgt=$2
    prefix=$3
    pushd $orig                
    tar zxvf ${download_path}/${prefix}$src-${tgt}.tgz
    popd 
}

generate_train(){
    lsrc=$1
    ltgt=$2
    src=${lsrc:0:2}    
    tgt=${ltgt:0:2}
    for ll in $lsrc $ltgt; do
        l=${ll:0:2}
        f="$orig/*/train.tags.$src-$tgt.$l"
        f_raw=$raw/train.$lsrc-$ltgt.$ll
        cat $f \
        | grep -v '<url>' \
        | grep -v '<talkid>' \
        | grep -v '<keywords>' \
        | grep -v '<speaker>' \
        | grep -v '<reviewer' \
        | grep -v '<translator' \
        | grep -v '<doc' \
        | grep -v '</doc>' \
        | sed -e 's/<title>//g' \
        | sed -e 's/<\/title>//g' \
        | sed -e 's/<description>//g' \
        | sed -e 's/<\/description>//g' \
        | sed 's/^\s*//g' \
        | sed 's/\s*$//g' \
        > $f_raw
        [ $? -eq 0 ] && echo "extracted $f to $f_raw"
    done
    return 0        
}

convert_valid_test(){
    src=$1
    tgt=$2
    for l in $src $tgt; do
        echo "lang: ${l}"
        for o in `ls $orig/*/IWSLT*.TED*.$src-$tgt.$l.xml`; do
            fname=${o##*/}
            f=$tmp/${fname%.*}
            echo "$o => $f"
            grep '<seg id' $o \
            | sed -e 's/<seg id="[0-9]*">\s*//g' \
            | sed -e 's/\s*<\/seg>\s*//g' \
            | sed -e "s/\’/\'/g" \
            > $f
            echo ""
        done
    done    
}

generate_subset(){
    lsrc=$1
    ltgt=$2
    src=${lsrc:0:2}
    tgt=${ltgt:0:2}
    subset=$3
    prefix=$4
    for ll in $lsrc $ltgt; do
        l=${ll:0:2}
        f=$tmp/$prefix.${src}-${tgt}.$l
        if [[ -f $f ]]; then        
            cp $f $raw/$subset.${lsrc}-$ltgt.${ll}
        fi
    done      
}
#################

echo "downloading iwslt training and dev data"
# using multilingual for it, nl 
download_iwslt "2017-01-trnmted" DeEnItNlRo DeEnItNlRo
download_iwslt "2017-01-trnted" ar en
download_iwslt "2017-01-trnted" en ar
download_iwslt "2017-01-trnted" ko en
download_iwslt "2017-01-trnted" en ko
download_iwslt "2015-01" vi en   
download_iwslt "2015-01" en vi   

echo "donwloading iwslt test data"
download_iwslt "2017-01-mted-test" it en "test."
download_iwslt "2017-01-mted-test" en it "test."
download_iwslt "2017-01-mted-test" nl en "test."
download_iwslt "2017-01-mted-test" en nl "test."

download_iwslt "2017-01-ted-test" ar en "test."
download_iwslt "2017-01-ted-test" en ar "test."
download_iwslt "2017-01-ted-test" ko en "test."
download_iwslt "2017-01-ted-test" en ko "test."
download_iwslt "2015-01-test" vi en "test."
download_iwslt "2015-01-test" en vi "test."

echo "extract training data tar balls"
extract_iwslt  DeEnItNlRo DeEnItNlRo
extract_iwslt  ar en
extract_iwslt  en ar
extract_iwslt  ko en
extract_iwslt  en ko
extract_iwslt  vi en   
extract_iwslt  en vi   


echo "extracting iwslt test data"
for lang in $langs; do
    l=${lang:0:2}
    extract_iwslt $l en "test."
    extract_iwslt en $l "test."
done

echo "convert dev and test data"
for lang in $langs; do
    s_lang=${lang:0:2}
    convert_valid_test $s_lang en  
    convert_valid_test en $s_lang
done



echo "creating training data into $raw"
for lang in $langs; do
    generate_train $lang en_XX
    generate_train en_XX $lang
done

echo "creating iwslt dev data into raw"
generate_subset en_XX vi_VN valid "IWSLT15.TED.tst2013"
generate_subset vi_VN en_XX valid "IWSLT15.TED.tst2013"

generate_subset en_XX ar_AR valid "IWSLT17.TED.tst2016"
generate_subset ar_AR en_XX valid "IWSLT17.TED.tst2016"
generate_subset en_XX ko_KR valid "IWSLT17.TED.tst2016"
generate_subset ko_KR en_XX valid "IWSLT17.TED.tst2016"


generate_subset en_XX it_IT valid "IWSLT17.TED.tst2010"
generate_subset it_IT en_XX valid "IWSLT17.TED.tst2010"
generate_subset en_XX nl_XX valid "IWSLT17.TED.tst2010"
generate_subset nl_XX en_XX valid "IWSLT17.TED.tst2010"

echo "creating iswslt test data into raw"
generate_subset en_XX vi_VN test "IWSLT15.TED.tst2015"
generate_subset vi_VN en_XX test "IWSLT15.TED.tst2015"

generate_subset en_XX ar_AR test "IWSLT17.TED.tst2017"
generate_subset ar_AR en_XX test "IWSLT17.TED.tst2017"
generate_subset en_XX ko_KR test "IWSLT17.TED.tst2017"
generate_subset ko_KR en_XX test "IWSLT17.TED.tst2017"

generate_subset en_XX it_IT test "IWSLT17.TED.tst2017.mltlng"
generate_subset it_IT en_XX test "IWSLT17.TED.tst2017.mltlng"
generate_subset en_XX nl_XX test "IWSLT17.TED.tst2017.mltlng"
generate_subset nl_XX en_XX test "IWSLT17.TED.tst2017.mltlng"

# normalze iwslt directions into x-en
pushd $raw
for lang in $langs; do
    for split in test valid; do
        x_en_f1=$split.$lang-en_XX.en_XX
        x_en_f2=$split.$lang-en_XX.${lang}

        en_x_f1=$split.en_XX-$lang.en_XX
        en_x_f2=$split.en_XX-$lang.${lang}        

        if [ -f $en_x_f1 ] && [ ! -f $x_en_f1 ]; then
            echo "cp $en_x_f1 $x_en_f1"
            cp $en_x_f1 $x_en_f1
        fi
        if [ -f $x_en_f2 ] && [ ! -f $x_en_f2 ]; then
            echo "cp $en_x_f2 $x_en_f2"
            cp $en_x_f2 $x_en_f2
        fi        
    done
done
popd