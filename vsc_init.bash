# need run in terminal
alias python=python3
alias pip=pip3
LS_ROOT=/content/drive/MyDrive/dataset/libri/root
SAVE_DIR=/content/drive/MyDrive/fairseq/s2t/libri/checkpoint
export PYTHONPATH=$PYTHONPATH:/content/drive/MyDrive/git/fairseq

#
COVOST_ROOT=/content/drive/MyDrive/dataset/cv/root_v4_en/

# aleady run
pip uninstall numpy
pip install --editable ./
pip install pandas torchaudio soundfile sentencepiece debugpy numpy


python3 -m examples.speech_to_text.prep_covost_data  --data-root ${COVOST_ROOT} --vocab-type char --src-lang en

```bash
clumba:
alias python=python3
alias pip=pip3
LS_ROOT=/content/drive/MyDrive/dataset/libri/root
SAVE_DIR=/content/drive/MyDrive/fairseq/s2t/libri/checkpoint
export PYTHONPATH=$PYTHONPATH:/content/drive/MyDrive/git/fairseq
pip install --editable ./
pip install pandas torchaudio soundfile sentencepiece debugpy
python -m examples.speech_to_text.prep_librispeech_data \
  --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
fairseq-train ${LS_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train-clean-100 --valid-subset test-clean \
  --num-workers 4 --max-tokens 40000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --share-decoder-input-output-embed \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 8

import os
print('clumba:-----:%s：%s'%(str(os.path.basename(__file__)).split('.')[0], str())) ##
```