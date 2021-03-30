[[Back]](..)

# S2T Example: Speech Recognition (ASR) on LibriSpeech
[LibriSpeech](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf) is a de-facto standard English ASR
benchmark. We provide competitive
vanilla [Transformer](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) baselines.

## Data preparation
Download and preprocess LibriSpeech data with
```bash
# additional Python packages for S2T data processing/model training
pip install pandas torchaudio sentencepiece

python examples/speech_to_text/prep_librispeech_data.py \
  --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
```
where `LS_ROOT` is the root path for downloaded data as well as generated files (manifest, features, vocabulary and
data configuration).

[Download](https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_vocab_unigram10000.zip) our vocabulary files
if you want to use our pre-trained models.

## Training
```bash
fairseq-train ${LS_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-tokens 40000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --share-decoder-input-output-embed \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 8
```
where `SAVE_DIR` is the checkpoint root path. Here we use `--arch s2t_transformer_s` (31M parameters) as example.
For better performance, you may switch to `s2t_transformer_m` (71M, with `--lr 1e-3`) or `s2t_transformer_l`
(268M, with `--lr 5e-4`). We set `--update-freq 8` to simulate 8 GPUs with 1 GPU. You may want to update it accordingly
when using more than 1 GPU.

## Inference & Evaluation
Average the last 10 checkpoints and evaluate on the 4 splits
(`dev-clean`, `dev-other`, `test-clean` and `test-other`):
```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
for SUBSET in dev-clean dev-other test-clean test-other; do
  fairseq-generate ${LS_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET} \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring wer
done
```

## Interactive Decoding
Launch the interactive console via
```bash
fairseq-interactive ${LS_ROOT} --config-yaml config.yaml --task speech_to_text \
  --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5
```
Type in WAV/FLAC/OGG audio paths (one per line) after the prompt.

## Results

| --arch | Params | dev-clean | dev-other | test-clean | test-other | Model |
|---|---|---|---|---|---|---|
| s2t_transformer_s | 30M | 3.8 | 8.9 | 4.4 | 9.0 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_s.pt) |
| s2t_transformer_m | 71M | 3.2 | 8.0 | 3.4 | 7.9 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_m.pt) |
| s2t_transformer_l | 268M | 3.0 | 7.5 | 3.2 | 7.5 | [Download](https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_l.pt) |

[[Back]](..)
