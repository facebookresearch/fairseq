# Speech-to-Text Translation

## CoVoST 2

### (Multilingual) Training
```bash
MANIFEST_ROOT=/private/home/changhan/data/datasets/speech/common_voice_20191210/fr_nl/manifests/ast_en_char
SAVE_DIR=/checkpoint/changhan/projects/fairseq_s2t/fr_nl_ast_en_char

mkdir ${SAVE_DIR}

python train.py ${MANIFEST_ROOT} --train-subset train_fr,train_nl --valid-subset dev_fr,dev_nl \
  --ddp-backend no_c10d --save-dir ${SAVE_DIR} --num-workers 4 --max-tokens 50000 --task speech_to_text \
  --criterion cross_entropy_acc --max-update 60000 --arch berard_512_3_2 --input-feat-per-channel 80 \
  --optimizer adam --lr 1e-3 --clip-norm 10.0 --lr-scheduler fixed \
  --log-format simple --log-interval 50 --keep-last-epochs 20 --seed 1 --bpe sentencepiece \
  --sentencepiece-vocab ${MANIFEST_ROOT}/spm_char.model --fp16 | tee -a "${SAVE_DIR}/train.log"

```

### Inference
```bash
MANIFEST_ROOT=/private/home/changhan/data/datasets/speech/common_voice_20191210/nl/manifests/ast_en_char
SAVE_DIR=/checkpoint/changhan/projects/fairseq_s2t/debug
python fairseq_cli/generate.py ${MANIFEST_ROOT} --task speech_to_text --path ${SAVE_DIR}/checkpoint_best.pt \
    --max-tokens 400000 --beam 5 --bpe sentencepiece --sentencepiece-vocab ${MANIFEST_ROOT}/spm_char.model --sacrebleu
```

## Self-Training

## Joint ASR+ST Training

