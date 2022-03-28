# Unit Language Model (ULM)

Here you can find links to the pre-trained ULMs and instructions on training new models using fairseq. At the end of the page, we also share how to run sampling for those models and provide pointers to the transcribed prompts we used.

## Pre-trained models

Using the links below, you can download pre-trained models for various unit types and vocabulary sizes:

| | 50 | 100 | 200
|-|-|-|-
| LogMel Filterbank | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/lm_km50/logmel50_lm.tgz)  |  [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/lm_km100/logmel100_lm.tgz) | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/lm_km200/logmel200_lm.tgz)
| Modified CPC | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/lm_km50/cpc50_lm.tgz) | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/lm_km100/cpc100_lm.tgz) | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/lm_km200/cpc200_lm.tgz)
| HuBERT | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/lm_km50/hubert50_lm.tgz) | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/lm_km100/hubert100_lm.tgz) | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/lm_km200/hubert200_lm.tgz)
| Wav2Vec 2.0 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/lm_km50/w2v2_50_lm.tgz) | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/lm_km100/w2v2_100_lm.tgz) | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/lm_km200/w2v2_200_lm.tgz)     


## Preprocessing data
Assuming that unit-transcribed train, valid, and test sets are located in `data/train.txt`, `data/valid.txt`, and `data/test.txt`, respectively,
we run the following command to get a preprocessed version of the datast in `data-bin`:

```bash
fairseq-preprocess --only-source \
        --trainpref data/train.txt --validpref data/valid.txt --testpref data/test.txt \
        --destdir data-bin/ --workers 40
```
As a result, the `data-bin` directory should appear.

## Fitting a Unit Language Model (ULM)
As an ULM, we train a standard fairseq Transformer LM. Assuming 8 GPUs used for training, a good starting point for an ULM training would be:
```bash
	fairseq-train data-bin/ \
        --task=language_modeling \
        --arch=transformer_lm_big \
        --share-decoder-input-output-embed \
        --dropout=0.1 \
        --attention-dropout=0.1 \
        --optimizer=adam \
        --adam-betas='(0.9, 0.98)' \
        --clip-norm=1.0 \
        --lr=0.0005 \
        --lr-scheduler=inverse_sqrt \
        --warmup-updates=4000 \
        --warmup-init-lr=1e-07 \
        --tokens-per-sample=3072 \
        --update-freq=16 \
        --max-tokens=4096 \
        --num-workers=4 \
        --skip-invalid-size-inputs-valid-test \
        --max-update=500000 \
        --log-interval=10 \
        --seed=100501 \
        --fp16 \
        --sample-break-mode=eos
```
This command will train a Transformer-large model (12 layers). You can train other standard LM models provided by fairseq, e.g. specify `--arch=transformer_lm` to train a smaller (6-layer) Transformer model. When training with a different number of GPUs, it might be a good idea to adjust the `update-freq` parameter. To save the GPU memory at an expense of additional computation, it can be useful to enable activation checkpointing with `--checkpoint-activations`.

## Sampling from an ULM
Once an ULM was trained, we can use it for generating new utterances. Suppose, that the prompts are given in a file named `prompts.txt`. Then we can sample continuations by running the following command:

```bash
    python sample.py  data-bin/ \
        --path=checkpoints/checkpoint_best.pt --task=language_modeling --sampling --temperature=0.7 \
        --seed=1  --prompts=prompts.txt  --output=samples.txt --max-len-a=0 --max-len-b=500 \
        --prefix-size=-1 --batch-size=16 --fp16 --samples-per-prompt=10
```
Here, `--prefix-size` controls the number of tokens that are used to prime the ULM. When set to a positive value, the sampling script will take first `prefix-size` tokens to prompt the ULM; with `0` it runs unconditional sampling and with `-1` the entire prompt is used. 
`--samples-per-prompt` specifies how many utterances are generated with every prompt which can be useful when generating multiple prompt continuations. In this command, `--max-len-a` and `--max-len-b` control the number of generated tokens. 

When using a pretrained model from above, `data-bin` should point to the unpacked directory (with `dict.txt` file).

Evaluation-time, to generate prompts, we used utterances from LibriSpeech dev-clean and test-clean that are longer than 6s. We took first 3s from an utterance as a prompt. Unit transcripts of those prompts can be downloaded here: [[dev]](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/dev_prompts.tgz) [[test]](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/test_prompts.tgz)

