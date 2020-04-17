# T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

## Original paper
https://arxiv.org/abs/1910.10683

## Introduction
The T5 model has been proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) paper by Colin Raffel et al. The original model was [implemented](https://github.com/google-research/text-to-text-transfer-transformer) in Tensoroflow Mesh. This reimplementation is based on the original work and the [reimplementation](https://github.com/huggingface/transformers) in HuggingFace Transformers.

## Pre-trained Models

Model | # params | Size | Download
------|---------:|-----:|:--------:
t5-small | 60M | 232MB | [link](https://applica-public.s3-eu-west-1.amazonaws.com/fairseq-t5/t5-small.pt)
t5-base | 220M | 853MB | [link](https://applica-public.s3-eu-west-1.amazonaws.com/fairseq-t5/t5-base.pt)
t5-large | 770M | 2.8GB | [link](https://applica-public.s3-eu-west-1.amazonaws.com/fairseq-t5/t5-large.pt)
t5-3B | 3B | 11GB | [link](https://applica-public.s3-eu-west-1.amazonaws.com/fairseq-t5/t5-3B.pt)
t5-11B |  11B | 43GB | [link](https://applica-public.s3-eu-west-1.amazonaws.com/fairseq-t5/t5-11B.pt)


All models use float32 and contain an optimizer state ([AdaFactor](https://arxiv.org/abs/1804.04235)).

## Finetuning
You can find [here](./trivia/README.md) a tutorial on finetuning on trivia questions.

~ ~ ~

The model was reimplemented at [Applica.ai](http://applica.ai/).