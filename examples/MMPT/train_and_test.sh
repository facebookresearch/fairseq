#!/bin/bash

model=$1
runs=runs/retri_v1_1/$model

mkdir -p $runs

echo "python locallaunch.py projects/retri/signclip_v1_1/$model.yaml --jobtype local_single > $runs/train.log 2>&1"
python locallaunch.py projects/retri/signclip_v1_1/$model.yaml --jobtype local_single > $runs/train.log 2>&1

echo "python locallaunch.py projects/retri/signclip_v1_1/test_$model.yaml --jobtype local_predict > $runs/test.log 2>&1"
python locallaunch.py projects/retri/signclip_v1_1/test_$model.yaml --jobtype local_predict > $runs/test.log 2>&1

echo "python locallaunch.py projects/retri/signclip_v1_1/test_${model}_zs.yaml --jobtype local_predict > $runs/test.zs.log 2>&1"
python locallaunch.py projects/retri/signclip_v1_1/test_${model}_zs.yaml --jobtype local_predict > $runs/test.zs.log 2>&1