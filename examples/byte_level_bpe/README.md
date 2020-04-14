# Neural Machine Translation with Byte-Level Subwords

https://arxiv.org/abs/1909.03341

We provide an implementation of byte-level byte-pair encoding (BBPE), taking IWSLT 2017 Fr-En translation as
example.

## Data
Get data and generate fairseq binary dataset:
```bash
bash ./get_data.sh
```

## Model Training
Train Transformer model with Bi-GRU embedding contextualization (implemented in `gru_transformer.py`):
```bash
# VOCAB=bytes
# VOCAB=chars
VOCAB=bbpe2048
# VOCAB=bpe2048
# VOCAB=bbpe4096
# VOCAB=bpe4096
# VOCAB=bpe16384
```
```bash
fairseq-train "data/bin_${VOCAB}" --task translation --user-dir examples/byte_level_bpe/gru_transformer \
    --arch gru_transformer --encoder-layers 2 --decoder-layers 2 --dropout 0.3 --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --log-format 'simple' --log-interval 100 --save-dir "checkpoints/${VOCAB}" \
    --max-sentences 100 --max-update 100000 --update-freq 2
```

## Generation
`fairseq-generate` requires bytes (BBPE) decoder to convert byte-level representation back to characters:
```bash
# BPE=--bpe bytes
# BPE=--bpe characters
BPE=--bpe byte_bpe --sentencepiece-model-path data/spm_bbpe2048.model
# BPE=--bpe sentencepiece --sentencepiece-vocab data/spm_bpe2048.model
# BPE=--bpe byte_bpe --sentencepiece-model-path data/spm_bbpe4096.model
# BPE=--bpe sentencepiece --sentencepiece-vocab data/spm_bpe4096.model
# BPE=--bpe sentencepiece --sentencepiece-vocab data/spm_bpe16384.model
```

```bash
fairseq-generate "data/bin_${VOCAB}" --task translation --user-dir examples/byte_level_bpe/gru_transformer \
    --source-lang fr --gen-subset test --sacrebleu --path "checkpoints/${VOCAB}/checkpoint_last.pt" \
    --tokenizer moses --moses-target-lang en ${BPE}
```
When using `fairseq-interactive`, bytes (BBPE) encoder/decoder is required to tokenize input data and detokenize model predictions:
```bash
fairseq-interactive "data/bin_${VOCAB}" --task translation --user-dir examples/byte_level_bpe/gru_transformer \
    --path "checkpoints/${VOCAB}/checkpoint_last.pt" --input data/test.fr --tokenizer moses --moses-source-lang fr \
    --moses-target-lang en ${BPE} --buffer-size 1000 --max-tokens 10000
```

## Results
| Vocabulary    | Model  | BLEU |
|:-------------:|:-------------:|:-------------:|
| Joint BPE 16k ([Kudo, 2018](https://arxiv.org/abs/1804.10959)) | 512d LSTM 2+2 | 33.81 |
| Joint BPE 16k | Transformer base 2+2 (w/ GRU) | 36.64 (36.72) |
| Joint BPE 4k | Transformer base 2+2 (w/ GRU) | 35.49 (36.10) |
| Joint BBPE 4k | Transformer base 2+2 (w/ GRU) | 35.61 (35.82) |
| Joint BPE 2k | Transformer base 2+2 (w/ GRU) | 34.87 (36.13) |
| Joint BBPE 2k | Transformer base 2+2 (w/ GRU) | 34.98 (35.43) |
| Characters | Transformer base 2+2 (w/ GRU) | 31.78 (33.30) |
| Bytes | Transformer base 2+2 (w/ GRU) | 31.57 (33.62) |


## Citation
```
@misc{wang2019neural,
    title={Neural Machine Translation with Byte-Level Subwords},
    author={Changhan Wang and Kyunghyun Cho and Jiatao Gu},
    year={2019},
    eprint={1909.03341},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


## Contact
Changhan Wang ([changhan@fb.com](mailto:changhan@fb.com)),
Kyunghyun Cho ([kyunghyuncho@fb.com](mailto:kyunghyuncho@fb.com)),
Jiatao Gu ([jgu@fb.com](mailto:jgu@fb.com))
