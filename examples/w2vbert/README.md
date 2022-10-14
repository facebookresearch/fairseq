# w2v-BERT

w2v-BERT learns speech representations from unlabeled data as described in [w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training (Chung et al., 2021)](https://arxiv.org/abs/2108.06209).

## Pre-trained models

Model | Finetuning split | Finetuning dataset | Checkpoint
|---|---|---|---
w2v-BERT 600M conformer - rel_pos (LV-60) | No finetuning | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/w2vbert/conformer/librilight/w2vbert_LL_en_relpos_600m.pt)
w2v-BERT 600M conformer - rel_pos (LV-60) | 100 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/w2vbert/conformer/librilight/w2vbert_LL_en_relpos_600m_LS100h_ft.pt)
w2v-BERT 600M conformer - rel_pos (LV-60) | 960 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/w2vbert/conformer/librilight/w2vbert_LL_en_relpos_600m_LS960h_ft.pt)
w2v-BERT 600M conformer - rope (LV-60) | No finetuning | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/w2vbert/conformer/librilight/w2vbert_LL_en_rope_600m.pt)
w2v-BERT 600M conformer - rope (LV-60) | 100 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/w2vbert/conformer/librilight/w2vbert_LL_en_rope_600m_LS100h_ft.pt)
w2v-BERT 600M conformer - rope (LV-60) | 960 hours | [Librispeech](http://www.openslr.org/12) | [download](https://dl.fbaipublicfiles.com/fairseq/w2vbert/conformer/librilight/w2vbert_LL_en_rope_600m_LS960h_ft.pt)

## Train a w2v-BERT model

The training procedure follows exactly the same as [wav2vec 2.0](https://github.com/fairinternal/fairseq-py/tree/ust/examples/wav2vec).  Specifically, to train a 600M conformer-based w2v-BERT model, run (you need to set `POS_ENC_TYPE` to one of `abs`, `rope`, and `rel_pos`, which decides which positional encoding method to use in the conformer layer):
```shell script
$ fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/w2vbert/config/pretraining \
    --config-name w2vbert_conformer_600m_librivox \
    --attn-type espnet --pos-enc-type ${POS_ENC_TYPE}
```

## Fine-tune a pre-trained model with CTC

The CTC fine-tuning procedure also follows exactly the same as [wav2vec 2.0](https://github.com/fairinternal/fairseq-py/tree/ust/examples/wav2vec#fine-tune-a-pre-trained-model-with-ctc).  Specifically, to fine-tune a pre-trained 600M conformer-based w2v-BERT model on 960h of LibriSpeech with letter targets, run:
```shell script
$ fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/w2vbert/config/finetuning \
    --config-name 600m_960h
```

The expected results (WER) are as follows (checkpoints that achieved the lowest WER on `dev-other` were selected for evaluation):
Model | Finetuning split | dev-clean | dev-other | test-clean | test-other
|---|---|---|---|---|---
w2v-BERT 600M conformer - rel_pos | 960h | 1.7 | 3.2 | 1.7 | 3.3 
w2v-BERT 600M conformer - rope | 960h | 1.7 | 3.1 | 1.7 | 3.5
w2v-BERT 600M conformer - rel_pos | 100h | 2.3 | 4.4 | 2.3 | 4.4
w2v-BERT 600M conformer - rope | 100h | 2.3 | 4.6 | 2.4 | 4.7
