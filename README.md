![No Language Left Behind](nllb.png?raw=true "NLLB")

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
| NLLB-200 | MoE | 54.5B |[model](https://tinyurl.com/nllb200moe54bmodel) | [metrics](https://tinyurl.com/nllb200moe54bmetrics), [translations](https://tinyurl.com/nllbflorestranslations) |
| NLLB-200 | Dense | 3.3B |[model](https://tinyurl.com/nllb200dense3bcheckpoint) | [metrics](https://tinyurl.com/nllb200dense3bmetrics) |
| NLLB-200 | Dense | 1.3B |[model](https://tinyurl.com/nllb200dense1bcheckpoint) | [metrics](https://tinyurl.com/nllb200dense1bmetrics) |
| NLLB-200-Distilled | Dense | 1.3B | [model](https://tinyurl.com/nllb200densedst1bcheckpoint) | [metrics](https://tinyurl.com/nllb200densedst1bmetrics) |
| NLLB-200-Distilled | Dense | 600M | [model](https://tinyurl.com/nllb200densedst600mcheckpoint) | [metrics](https://tinyurl.com/nllb200densedst600mmetrics) |

All models are licensed under CC-BY-NC 4.0 available in [Model LICENSE](LICENSE.model.md) file. We provide FLORES-200 evaluation results for all the models. For more details see the [Modeling section README](examples/nllb/modeling/README.md).

:star: NEW :star: : We are releasing all the translations of NLLB-200 MoE model. Check [Evaluation section README](examples/nllb/evaluation/README.md) for more details.


> Please use `wget --trust-server-names <url>` to download the provided links in proper file format.

### LID Model
LID (**L**anguage **ID**entification) model to predict the language of the input text is available [here](https://tinyurl.com/nllblid218e) under [CC-BY-NC 4.0](LICENSE.model.md) license.


### LASER3 Encoder Models
LASER3 models are available at [LASER](https://github.com/facebookresearch/LASER).


### HuggingFace Integrations

Support for the dense models is available through the Hugging Face Hub under the [`NLLB`](https://huggingface.co/models?other=nllb) tag. It is supported in the `transformers` library and the documentation for this model is available [here](https://huggingface.co/docs/transformers/main/en/model_doc/nllb#nllb).

Input and output languages are entirely customizable with BCP-47 codes used by the FLORES-200 dataset, here's an example usage with a translation from Romanian to German:

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="ron_Latn")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True)

>>> article = "Şeful ONU spune că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
```
Result:
```
UN-Chef sagt, es gibt keine militärische Lösung in Syrien
```

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

## Human Evaluations - (XSTS)
(Added Jan - 2023) We open-source additional guidelines and training materials for conducting the human evaluation protocol we utilized (XSTS), as well the calibration utilized and the entire human translation evaluation set for NLLB-200 and it's published baseline [here](examples/nllb/human_XSTS_eval/README.md).

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
