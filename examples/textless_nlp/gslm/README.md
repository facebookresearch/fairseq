# Generative Spoken Language Modeling

* [Paper](https://arxiv.org/abs/2102.01192)
* [Demo](https://speechbot.github.io/gslm/index.html)

We build and evaluate generative speech2speech systems using [Log Mel Filtebank](https://pytorch.org/audio/stable/compliance.kaldi.html#fbank), [Modified CPC](https://github.com/facebookresearch/CPC_audio), [HuBERT Base](https://github.com/pytorch/fairseq/tree/master/examples/hubert) and [Wav2Vec 2.0 Large](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec). Our system is composed of three components, namely, *speech2unit*, *ulm* and *unit2speech*. We explain about models and usage of these components in their respective sub-directories. See the links below.

## Speech to Unit Model (speech2unit)
Speech to unit model is used for quantizing raw speech into learned discrete speech units. [More details](speech2unit)

## Unit Language Model (ulm)
Unit Language Model is a generative language model trained on discrete speech units. [More details](ulm)

## Unit to Speech Model (unit2speech)
Unit to speech model is used for synthesizing speech from discrete speech units. [More details](unit2speech)

## Metrics
We show how to compute ASR based metrics as well as zero-shot metrics proposed in our paper [here](metrics).

## Tools
We share two tools to resynthesize a given spoken utterance, and generate novel spoken language given a spoken prompt. [More detail](tools)