#!/bin/bash

nvidia-smi
pip install -r requirements.txt
python3 main_mt.py -m transformer -b 128  -d EN_FR -g 0 -layer 3 -s dgx #> energyLM-wiki2.out
