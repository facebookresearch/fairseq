<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://opensource.fb.com/support-ukraine"><img alt="Support Ukraine" src="https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB" /></a>
  <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p>

--------------------------------------------------------------------------------

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.

We provide reference implementations of various sequence modeling papers:

<details><summary>List of implemented papers</summary><p>

* **Convolutional Neural Networks (CNN)**
  + [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)
  + [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
  + [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  + [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
  + [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* **LightConv and DynamicConv models**
  + [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* **Long Short-Term Memory (LSTM) networks**
  + Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
* **Transformer (self-attention) networks**
  + Attention Is All You Need (Vaswani et al., 2017)
  + [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
  + [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
  + [Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)](examples/language_model/README.adaptive_inputs.md)
  + [Lexically constrained decoding with dynamic beam allocation (Post & Vilar, 2018)](examples/constrained_decoding/README.md)
  + [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., 2019)](examples/truncated_bptt/README.md)
  + [Adaptive Attention Span in Transformers (Sukhbaatar et al., 2019)](examples/adaptive_span/README.md)
  + [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
  + [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
  + [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
  + [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
  + [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
  + [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
  + [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
  + [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
  + [Generating Medical Reports from Patient-Doctor Conversations Using Sequence-to-Sequence Models (Enarvi et al., 2020)](examples/pointer_generator/README.md)
  + [Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)](examples/linformer/README.md)
  + [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
  + [Deep Transformers with Latent Depth (Li et al., 2020)](examples/latent_depth/README.md)
  + [Unsupervised Cross-lingual Representation Learning for Speech Recognition (Conneau et al., 2020)](https://arxiv.org/abs/2006.13979)
  + [Self-training and Pre-training are Complementary for Speech Recognition (Xu et al., 2020)](https://arxiv.org/abs/2010.11430)
  + [Robust wav2vec 2.0: Analyzing Domain Shift in Self-Supervised Pre-Training (Hsu, et al., 2021)](https://arxiv.org/abs/2104.01027)
  + [Unsupervised Speech Recognition (Baevski, et al., 2021)](https://arxiv.org/abs/2105.11084)
  + [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition (Xu et al., 2021)](https://arxiv.org/abs/2109.11680)
  + [VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding (Xu et. al., 2021)](https://arxiv.org/pdf/2109.14084.pdf)
  + [VLM: Task-agnostic Video-Language Model Pre-training for Video Understanding (Xu et. al., 2021)](https://aclanthology.org/2021.findings-acl.370.pdf)
  + [NormFormer: Improved Transformer Pretraining with Extra Normalization (Shleifer et. al, 2021)](examples/normformer/README.md)
* **Non-autoregressive Transformers**
  + Non-Autoregressive Neural Machine Translation (Gu et al., 2017)
  + Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement (Lee et al. 2018)
  + Insertion Transformer: Flexible Sequence Generation via Insertion Operations (Stern et al. 2019)
  + Mask-Predict: Parallel Decoding of Conditional Masked Language Models (Ghazvininejad et al., 2019)
  + [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* **Finetuning**
  + [Better Fine-Tuning by Reducing Representational Collapse (Aghajanyan et al. 2020)](examples/rxf/README.md)

</p></details>

### What's New:
* December 2021 [Released Direct speech-to-speech translation code](examples/speech_to_speech/README.md)
* October 2021 [Released VideoCLIP and VLM models](examples/MMPT/README.md)
* October 2021 [Released multilingual finetuned XLSR-53 model](examples/wav2vec/README.md)
* September 2021 [`master` branch renamed to `main`](https://github.com/github/renaming).
* July 2021 [Released DrNMT code](examples/discriminative_reranking_nmt/README.md)
* July 2021 [Released Robust wav2vec 2.0 model](examples/wav2vec/README.md)
* June 2021 [Released XLMR-XL and XLMR-XXL models](examples/xlmr/README.md)
* May 2021 [Released Unsupervised Speech Recognition code](examples/wav2vec/unsupervised/README.md)
* March 2021 [Added full parameter and optimizer state sharding + CPU offloading](examples/fully_sharded_data_parallel/README.md)
* February 2021 [Added LASER training code](examples/laser/README.md)
* December 2020: [Added Adaptive Attention Span code](examples/adaptive_span/README.md)
* December 2020: [GottBERT model and code released](examples/gottbert/README.md)
* November 2020: Adopted the [Hydra](https://github.com/facebookresearch/hydra) configuration framework
  * [see documentation explaining how to use it for new and existing projects](docs/hydra_integration.md)
* November 2020: [fairseq 0.10.0 released](https://github.com/pytorch/fairseq/releases/tag/v0.10.0)
* October 2020: [Added R3F/R4F (Better Fine-Tuning) code](examples/rxf/README.md)
* October 2020: [Deep Transformer with Latent Depth code released](examples/latent_depth/README.md)
* October 2020: [Added CRISS models and code](examples/criss/README.md)

<details><summary>Previous updates</summary><p>

* September 2020: [Added Linformer code](examples/linformer/README.md)
* September 2020: [Added pointer-generator networks](examples/pointer_generator/README.md)
* August 2020: [Added lexically constrained decoding](examples/constrained_decoding/README.md)
* August 2020: [wav2vec2 models and code released](examples/wav2vec/README.md)
* July 2020: [Unsupervised Quality Estimation code released](examples/unsupervised_quality_estimation/README.md)
* May 2020: [Follow fairseq on Twitter](https://twitter.com/fairseq)
* April 2020: [Monotonic Multihead Attention code released](examples/simultaneous_translation/README.md)
* April 2020: [Quant-Noise code released](examples/quant_noise/README.md)
* April 2020: [Initial model parallel support and 11B parameters unidirectional LM released](examples/megatron_11b/README.md)
* March 2020: [Byte-level BPE code released](examples/byte_level_bpe/README.md)
* February 2020: [mBART model and code released](examples/mbart/README.md)
* February 2020: [Added tutorial for back-translation](https://github.com/pytorch/fairseq/tree/main/examples/backtranslation#training-your-own-model-wmt18-english-german)
* December 2019: [fairseq 0.9.0 released](https://github.com/pytorch/fairseq/releases/tag/v0.9.0)
* November 2019: [VizSeq released (a visual analysis toolkit for evaluating fairseq models)](https://facebookresearch.github.io/vizseq/docs/getting_started/fairseq_example)
* November 2019: [CamemBERT model and code released](examples/camembert/README.md)
* November 2019: [BART model and code released](examples/bart/README.md)
* November 2019: [XLM-R models and code released](examples/xlmr/README.md)
* September 2019: [Nonautoregressive translation code released](examples/nonautoregressive_translation/README.md)
* August 2019: [WMT'19 models released](examples/wmt19/README.md)
* July 2019: fairseq relicensed under MIT license
* July 2019: [RoBERTa models and code released](examples/roberta/README.md)
* June 2019: [wav2vec models and code released](examples/wav2vec/README.md)

</p></details>

### Features:

* multi-GPU training on one machine or across multiple machines (data and model parallel)
* fast generation on both CPU and GPU with multiple search algorithms implemented:
  + beam search
  + Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  + sampling (unconstrained, top-k and top-p/nucleus)
  + [lexically constrained decoding](examples/constrained_decoding/README.md) (Post & Vilar, 2018)
* [gradient accumulation](https://fairseq.readthedocs.io/en/latest/getting_started.html#large-mini-batch-training-with-delayed-updates) enables training with large mini-batches even on a single GPU
* [mixed precision training](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-with-half-precision-floating-point-fp16) (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
* [extensible](https://fairseq.readthedocs.io/en/latest/overview.html): easily register new models, criterions, tasks, optimizers and learning rate schedulers
* [flexible configuration](docs/hydra_integration.md) based on [Hydra](https://github.com/facebookresearch/hydra) allowing a combination of code, command-line and file based configuration
* [full parameter and optimizer state sharding](examples/fully_sharded_data_parallel/README.md)
* [offloading parameters to CPU](examples/fully_sharded_data_parallel/README.md)

We also provide [pre-trained models for translation and language modeling](#pre-trained-models-and-examples)
with a convenient `torch.hub` interface:

``` python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```

# No Language Left Behind
No Language Left Behind (NLLB) is a first-of-its-kind, AI breakthrough project that open-sources models capable of delivering high-quality translations directly between any pair of 200+ languages — including low-resource languages like Asturian, Luganda, Urdu and more. It aims to help people communicate with anyone, anywhere, regardless of their language preferences.

To enable the community to leverage and build on top of NLLB, we open source all our evaluation benchmarks(FLORES-200, NLLB-MD, Toxicity-200), LID models and training code, LASER3 encoders, data mining code, MMT training and inference code and our final NLLB-200 models and their smaller distilled versions, for easier use and adoption by the research community.

This code repository contains instructions to get the datasets, optimized training and inference code for MMT models, training code for LASER3 encoders as well as instructions for downloading and using the final large NLLB-200 model and the smaller distilled models.
In addition to supporting more than 200x200 translation directions, we also provide reliable evaluations of our model on all possible translation directions on the FLORES-200 benchmark. By open-sourcing our code, models and evaluations, we hope to foster even more research in low-resource languages leading to further improvements in the quality of low-resource translation through contributions from the research community.



- [Paper](https://research.facebook.com/publications/no-language-left-behind/)
- [Website](https://ai.facebook.com/research/no-language-left-behind/)
- [Blog](https://ai.facebook.com/blog/nllb-200-high-quality-machine-translation/)
- [NLLB Demo](https://nllb.metademolab.com/)
- [Request for Proposals : Translation Support for African Languages](https://ai.facebook.com/research/request-for-proposals/translation-support-for-african-languages/)



## Open Sourced Models and Community Integrations

### Multilingual Translation Models
| Model Name | Model Type | #params | checkpoint | metrics |
| - | - | - | - | - |
| NLLB-200 | MoE | 54.5B |[model](https://tinyurl.com/nllb200moe54bmodel) | [metrics](https://tinyurl.com/nllb200moe54bmetrics) |
| NLLB-200 | Dense | 3.3B |[model](https://tinyurl.com/nllb200dense3bcheckpoint) | [metrics](https://tinyurl.com/nllb200dense3bmetrics) |
| NLLB-200 | Dense | 1.3B |[model](https://tinyurl.com/nllb200dense1bcheckpoint) | [metrics](https://tinyurl.com/nllb200dense1bmetrics) |
| NLLB-200-Distilled | Dense | 1.3B | [model](https://tinyurl.com/nllb200densedst1bcheckpoint) | [metrics](https://tinyurl.com/nllb200densedst1bmetrics) |
| NLLB-200-Distilled | Dense | 600M | [model](https://tinyurl.com/nllb200densedst600mcheckpoint) | [metrics](https://tinyurl.com/nllb200densedst600mmetrics) |

All models are licensed under CC-BY-NC 4.0 available in [Model LICENSE](LICENSE.model.md) file. We provide FLORES-200 evaluation results for all the models. For more details see the [Modeling section README](examples/nllb/modeling/README.md).

> Please use `wget --trust-server-names <url>` to download the provided links in proper file format.

### LID Model
LID (**L**anguage **ID**entification) model to predict the language of the input text is available [here](https://tinyurl.com/nllblid218e) under [CC-BY-NC 4.0](LICENSE.model.md) license.


### LASER3 Encoder Models
LASER3 models are available at [LASER](https://github.com/facebookresearch/LASER).


### HuggingFace Integrations
[Coming Soon]


## Installation
Follow installation instructions in [INSTALL](INSTALL.md) guide for running training/generation. For general instructions about `fairseq` and working with the codebase refer to [`fairseq` README](https://github.com/facebookresearch/fairseq). For [stopes](https://github.com/facebookresearch/stopes) and [LASER](https://github.com/facebookresearch/LASER) follow their README files for installation.

## Datasets
NLLB project uses data from three sources : public bitext, mined bitext and data generated using backtranslation. Details of different datasets used and open source links are provided in details [here](examples/nllb/data/README.md).

### Primary Bitext
We provide a download script for [public bitext data](examples/nllb/data/README.md), and links to download [NLLB-Seed](https://github.com/facebookresearch/flores/tree/main/nllb_seed) data. For more details check [here](examples/nllb/data/README.md).

### Mined Bitext
LASER3 teacher-student training code is open sourced [here](examples/nllb/laser_distillation/README.md). LASER3 encoders and mined bitext metadata are open sourced in [LASER](https://github.com/facebookresearch/LASER) repository.
Global mining pipeline and monolingual data filtering pipelines are released and available in our [stopes](https://github.com/facebookresearch/stopes) repository.

### Backtranslated Bitext
Follow the instructions [here](examples/nllb/data/README.md) to generate backtranslated data from a pretrained model.

### Preparing Datasets for Training
We open source our dataset preparation pipeline for filtering/encoding/binarizing large scale datasets in [stopes](https://github.com/facebookresearch/stopes). Encoding the datasets are done using the new `SPM-200` model which was trained on 200+ languages used in the NLLB project. For more details see [link](examples/nllb/modeling/README.md).

| SPM-200 Artifacts | download links |
| - | - |
| Model | [link](https://tinyurl.com/flores200sacrebleuspm) |
| Dictionary| [link](https://tinyurl.com/nllb200dictionary) |


## Training NLLB Models
We open source all our model training and generation code in this repo. We also share code for finetuning our models on different domains like NLLB-MD. Additionally, we also share the code for online distillation that produced our 1.3B and 600M distilled models. For more details check the [Modeling section Readme](examples/nllb/modeling/README.md).

## Evaluation and Generation
NLLB project includes release of evaluation datasets like Flores-200, NLLB-MD and Toxicity-200. For instructions to run evaluation see instructions [here](https://github.com/facebookresearch/flores/tree/main/flores200) and for instructions to produce generations from the models follow instructions [here](examples/nllb/modeling#generationevaluation).

[Flores200](https://github.com/facebookresearch/flores/tree/main/flores200) |
[NLLB-MD](https://github.com/facebookresearch/flores/tree/main/nllb_md) |
[Toxicity-200](https://github.com/facebookresearch/flores/tree/main/toxicity)

## Citation
If you use NLLB in your work or any models/datasets/artifacts published in NLLB, please cite :

```bibtex
@article{nllb2022,
  title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  author={{NLLB Team} and Costa-jussà, Marta R. and Cross, James and Çelebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and Sun, Anna and Wang, Skyler and Wenzek, Guillaume and Youngblood, Al and Akula, Bapi and Barrault, Loic and Mejia-Gonzalez, Gabriel and Hansanti, Prangthip and Hoffman, John and Jarrett, Semarley and Sadagopan, Kaushik Ram and Rowe, Dirk and Spruit, Shannon and Tran, Chau and Andrews, Pierre and Ayan, Necip Fazil and Bhosale, Shruti and Edunov, Sergey and Fan, Angela and Gao, Cynthia and Goswami, Vedanuj and Guzmán, Francisco and Koehn, Philipp and Mourachko, Alexandre and Ropers, Christophe and Saleem, Safiyyah and Schwenk, Holger and Wang, Jeff},
  year={2022}
}
```

## License
NLLB code and fairseq(-py) is MIT-licensed available in [LICENSE](LICENSE) file.
