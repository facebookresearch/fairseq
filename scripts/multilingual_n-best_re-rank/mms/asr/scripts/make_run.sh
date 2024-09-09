wavs=$1
lids=$2
model=$3
dst=$4

echo "#!/bin/bash" > $dst/run.sh
python scripts/make_run.sh --wavs $wavs --lids $lids --model $model --dst $dst
chmod +x $dst/run.sh