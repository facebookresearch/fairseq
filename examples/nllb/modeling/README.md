# No Language Left Behind : Modeling

## Introduction

This README contains details of how to use the multilingual machine translation models we trained for NLLB-200. These neural multilingual machine translation models support translation between any pair of [200 languages](scripts/flores200/langs.txt). Not only do we support more than 200x200 translation directions, we also release the [quality](#open-sourced-models-and-metrics) of our final model on each of the 200x200 translation directions by measuring spBLEU and chrf++ on the [FLORES-200 benchmark](https://github.com/facebookresearch/flores). In the sections below, we share where our models can be downloaded, how they can be used for inference or finetuning, and more details about how to train your own multilingual machine translation models with Sparsely Gated Mixture of Experts, Expert Output Masking, Curriculum Learning, incorporating self-supervised learning (denoising auto-encoder like mBART), Online Distillation. We also share the training recipe for our final NLLB-200 MoE-128 model.

## Citing Our Work

If you use any of the code or models from this repo please cite the following paper :

```bibtex
@article{nllb2022,
  title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  author={{NLLB Team} and Costa-jussà, Marta R. and Cross, James and Çelebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and Sun, Anna and Wang, Skyler and Wenzek, Guillaume and Youngblood, Al and Akula, Bapi and Barrault, Loic and Mejia-Gonzalez, Gabriel and Hansanti, Prangthip and Hoffman, John and Jarrett, Semarley and Sadagopan, Kaushik Ram and Rowe, Dirk and Spruit, Shannon and Tran, Chau and Andrews, Pierre and Ayan, Necip Fazil and Bhosale, Shruti and Edunov, Sergey and Fan, Angela and Gao, Cynthia and Goswami, Vedanuj and Guzmán, Francisco and Koehn, Philipp and Mourachko, Alexandre and Ropers, Christophe and Saleem, Safiyyah and Schwenk, Holger and Wang, Jeff},
  year={2022}
}
```


## Open Sourced Models and Metrics

| Model Name | Model Type | arch | #params | checkpoint | metrics |
| - | - | - | - | - | - |
| NLLB-200 | MoE-128 | transformer_24_24_big | 54.5B |[model](https://tinyurl.com/nllb200moe54bmodel) | [metrics](https://tinyurl.com/nllb200moe54bmetrics) |
| NLLB-200 | Dense | transformer_24_24_big | 3.3B |[model](https://tinyurl.com/nllb200dense3bcheckpoint) | [metrics](https://tinyurl.com/nllb200dense3bmetrics) |
| NLLB-200 | Dense | transformer_24_24 | 1.3B |[model](https://tinyurl.com/nllb200dense1bcheckpoint) | [metrics](https://tinyurl.com/nllb200dense1bmetrics) |
| NLLB-200-Distilled | Dense | transformer_24_24 | 1.3B | [model](https://tinyurl.com/nllb200densedst1bcheckpoint) | [metrics](https://tinyurl.com/nllb200densedst1bmetrics) |
| NLLB-200-Distilled | Dense | transformer_12_12 | 600M | [model](https://tinyurl.com/nllb200densedst600mcheckpoint) | [metrics](https://tinyurl.com/nllb200densedst600mmetrics) |

All models are licensed under CC-BY-NC 4.0 available in [Model LICENSE](../../../LICENSE.model.md) file. We also provide a `metrics.csv` file for each model. For the NLLB-200 54B MoE model, we share the results on all 40602 translation directions of Flores-200. Results contain `chrF++`, `spBLEU(SPM-200)`, `spBLEU(SPM-100)`(using Flores-101 tokenizer). For the other models we provide the evaluation results(only `chrF++`) on a sampled set of directions (all possible 402 English-centric direction + 200 sampled non English-centric directions).

NLLB-200(MoE, 54.5B) model has 128 experts but was trained on 256 GPUs. For this model, the `replication_count` = 256/128 = 2. See section [Generation for MoE](#moe-2) for more details.

To download a model or metrics file :
> Please use `wget --trust-server-names <url>` to download the provided links in proper file format.


## Prerequisites

### Installation

Please follow the instructions for installation [here](). Our training and evaluation pipelines are also dependent on the [`stopes`](https://github.com/facebookresearch/stopes) library.

<details><summary>Additional Information</summary>
<p>
With Hydra, options are configured via [Hydra](https://hydra.cc/docs/intro/), and used with configuration files are provided:


* Training: [`train/conf/cfg/default.yaml`](train/conf/cfg/default.yaml)
* Evaluation:  [`evaluation/conf/generate_multi.yaml`](evaluation/conf/generate_multi.yaml)

</p>
</details>


### Filtering and Preparing the Data

Data preparation is managed by the [stopes](https://github.com/facebookresearch/stopes) library. Specifically:

1. Data filtering is performed using `stopes` corpus filtering pipeline. Please consult the filtering pipeline’s README file and example configuration for more details.
2. Once filtered, data can then be preprocessed/binarized with the prepare_data pipeline in `stopes`. Please consult the README file for further information.


## Training various NLLB models

We provide scripts for training the NLLB-200 model and other variants. They launch training using train_script.py, which is configured using Hydra (similar to generation/evaluation). The default configuration can be found at [`train/conf/cfg/default.yaml`](train/conf/cfg/default.yaml), and the training configs we include are `flores_200_full`, `flores_200_full_moe`, `flores_200_ablation`, `flores_200_ablation_moe`, and `flores_200_ablation_ssl`. You can specify the config as `cfg=flores_200_full` in the `train_script.py ...` command to train with a particular configuration.

Additionally update the following:

* [`train/conf/cfg/cluster/example.yaml`](train/conf/cfg/cluster/example.yaml) with the partition of your SLURM cluster
* `train/conf/cfg/dataset/*.yaml` with the data_prefix (and `mono_data_prefix` if using SSL) of your encoded/binarized data path
* Set `$OUTPUT_DIR` to the output directory for checkpoints and training logs
* Set `$DROP` to the overall dropout rate

#### Dense

```bash
python examples/nllb/modeling/train/train_script.py \
    cfg=flores_200_full \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=$OUTPUT_DIR \
    cfg.dataset.eval_lang_pairs_file=examples/nllb/modeling/scripts/flores200/eval_lang_pairs_eng400_noneng20.txt \
    cfg.dropout=$DROP
```

#### MoE

```bash
python examples/nllb/modeling/train/train_script.py \
    cfg=flores_200_full_moe \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=$OUTPUT_DIR \
    cfg.dataset.eval_lang_pairs_file=examples/nllb/modeling/scripts/flores200/eval_lang_pairs_eng400_noneng20.txt \
    cfg.dropout=$DROP \
    cfg.model_type.expert_count=128 \
    cfg.model_type.moe_param=" --moe --moe-freq $MOE_FREQ "
```

#### Dense Bilingual

An example bilingual model for `eng_Latn-npi_Deva` can be trained with the following command:

```bash
python examples/nllb/modeling/train/train_script.py \
    cfg=bilingual \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=$OUTPUT_DIR \
    cfg.dropout=$DROP
```

### Training with Mixture of Experts Optimizations

#### Training with Expert Output Masking

To add Expert Output Masking, use the training command from MoE, with  `cfg.model_type.moe_param=" --moe --moe-freq $MOE_FREQ --moe-eom $MOE_EOM "`


#### Training with Conditional MoE Routing

To add Conditional MoE Routing, use the training command from MoE, with `cfg.model_type.moe_param=" --moe --moe-freq $MOE_FREQ --moe-cmr --cmr-gate-drop $CMR_GATE_DROP "`

### Training with a Curriculum

Use the following parameters for training with either Naive or Phased Curriculum approaches:

```bash
cfg.restore_file=$RESTORE_FILE
cfg.max_updates=$MAX_UPDATES
# Set this so that there is a consistent save_dir across all phases of the curriculum
cfg.max_update_str=$MAX_UPDATE_STR
cfg.resume_finished=true
```

### Training with SSL objectives

Monolingual data can be incorporated into training using one SSL objectives by specifying one of the following  values to the ssl_task training configuration option:

* mono_dae: mBART-style denoising objective
* mono_lm: left-to-right language model objective on the decoder side (dummy encoder input)
* mono_mixed_task: monolingual examples probabilistically split between the above (p=0.5)

In order to use SSL objectives for training, binarized monolingual data needs to be provided by specifying the mono_num_shards and mono_data_prefix options in the dataset config. Note that we found the first of these options (mono_dae) helpful for smaller models, and in particular for training back-translation models, but SSL objectives did not provide additional benefits for the full model when applied to the same monolingual data that had been used for back-translation.

## Training the NLLB-200 MoE-128 model

```bash
python examples/nllb/modeling/train/train_script.py \
    cfg=flores_200_full_moe \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=$OUTPUT_DIR \
    cfg.max_update_str="mu170.230.270" \
    cfg.max_updates=170000 \
    cfg.dataset.lang_pairs_file=examples/nllb/modeling/scripts/flores200/final_lang_pairs_cl3.txt \
    cfg.lr=0.002 \
    cfg.resume_finished=true \
    cfg.dropout=0.3 \
    cfg.model_type.expert_count=128 \
    cfg.model_type.moe_param=" --moe --moe-freq 4 --moe-eom 0.2 "
```


| update step | max_updates | lang_pairs_file | restore_file | reset_dataloader |
| - | - | - | - | - |
| 0 | 170000 | final_lang_pairs_cl3.txt | ~ | false |
| 170000   | 230000 | final_lang_pairs_cl2.txt | $OUTPUT_DIR/checkpoint_17_170000.pt | false |
| 230000   | 270000 | final_lang_pairs_cl1.txt | $OUTPUT_DIR/checkpoint_22_230000.pt | false |
| 270000   | 300000 | lang_pairs.txt | $OUTPUT_DIR/checkpoint_25_270000.pt | true |

## Online Distillation

To perform online distillation from any models trained above, set the following parameters:


```bash
# transformer_24_24 is the 1.3B parameter setting, use transformer_12_12 for the 615M setting
cfg.arch="transformer_24_24"
# Set the soft loss weight
cfg.alpha_ce=$ALPHA_CE
cfg.kd_temp=1.0
cfg.teacher_path=$TEACHER_PATH
```

## Finetuning NLLB models

Like evaluation data, finetuning data should also be bitexts in languages supported by NLLB-200. First, you need to:

* Prepare the data you want to finetune NLLB-200 on.
* Set `$DATA_CONFIG` to a dataset config file (see [Preparing the Data]()).
* Set `$OUTPUT_DIR` to the output directory for checkpoints and training logs.
* Set `$MODEL_FOLDER` to the folder with the downloaded NLLB-200 model checkpoints.
* Set `$DROP` to the overall dropout rate.
* Set `$SRC`, `$TGT` to the source and target languages.

You can finetune NLLB-200 on the bilingual data (from $src to $tgt) with the following commands depending on whether you’re finetuning a dense (3.3B) or an MoE model:

#### Dense

```bash
DROP=0.1
python examples/nllb/modeling/train/train_script.py \
    cfg=nllb200_dense3.3B_finetune_on_fbseed \
    cfg/dataset=$DATA_CONFIG \
    cfg.dataset.lang_pairs="$SRC-$TGT" \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=$OUTPUT_DIR \
    cfg.dropout=$DROP \
    cfg.warmup=10 \
    cfg.finetune_from_model=$MODEL_FOLDER/checkpoint.pt

```

### MoE

```bash
DROP=0.3
python examples/nllb/modeling/train/train_script.py \
    cfg=nllb200_moe_finetune_on_fbseed \
    cfg/dataset=$DATA_CONFIG \
    cfg.dataset.lang_pairs="$SRC-$TGT" \
    cfg.fairseq_root=$(pwd) \
    cfg.output_dir=$OUTPUT_DIR \
    cfg.dropout=$DROP \
    cfg.model_type.moe_param=" --moe --moe-freq 4 --moe-eom 0.2 --moe-gate-loss-wt 0 " \
    cfg.model_type.expert_count=128 \
    cfg.warmup=10 \
    cfg.reset_all=true \
    cfg.restore_file=$MODEL_FOLDER/checkpoint_2_300000.pt
```
## Generation/Evaluation

### Prerequisites

1. Follow the steps to download SPM-200 and FLORES-200 dataset [here](https://github.com/facebookresearch/flores/flores200).
2. Requires `sacrebleu>=2.1.0`
3. Update [`evaluation/conf/cluster/example.yaml`](evaluation/conf/cluster/example.yaml) with your `data_dir`, `flores_path`, `cluster` partition, and `non_flores_path` (for non-FLORES evaluation).
4. We use and recommend using `chrF++` metric which is automatically calculated using `sacrebleu` in our evaluation pipeline.

### Dense

The following is an example generation/evaluation command launched via the generate_multi.py script, where `$MODEL_FOLDER` specifies the training directory where checkpoint files are located. As with full training commands, configuration is specified in a composable fashion using [Hydra](https://hydra.cc/docs/intro/).

In the case of this command, the base configuration can be found in the file [`generate_multi_full.yaml`](evaluation/conf/generate_multi_full.yaml), specified by supplying the name `generate_multi_full` as the `--config-name` argument. The remainder of the command-line arguments selectively override individual configuration settings. See [`generate_multi.yaml`](evaluation/conf/generate_multi.yaml) for a fully-specified config.

Settings indicated "???" in config files (`model_folder` and `fairseq_root`) are required to be specified on the command line. `fairseq_root` is the path to the fairseq repo, which is specified as the output of `pwd` here because it is presumed to be the directory from which this command is launched.

```bash
python examples/nllb/modeling/evaluation/generate_multi.py \
  model_folder=$MODEL_FOLDER \
  checkpoints="[checkpoint_27_200000]" \
  fairseq_root=`pwd` \
  lang_pairs_per_job=10 \
  encoder_langtok=src \
  model_type=dense \
  lang_config=flores200.sampled \
  eval_on=all \
  gen_splits="[valid]" \
  --config-name generate_multi_full
```


### MoE

`replication_count` needs to be set to `num_training_gpus/num_experts` when `num_training_gpus > num_experts`. For example, the NLLB-200 MoE 54.5B model has 128 experts but was trained on 256 GPUs. So for that model `replication_count` = 256/128 = 2.

```bash
python examples/nllb/modeling/evaluation/generate_multi.py \
  model_folder=$MODEL_FOLDER \
  checkpoints="[checkpoint_27_200000]" \
  fairseq_root=`pwd` \
  lang_pairs_per_job=10 \
  encoder_langtok=src \
  replication_count=2 \
  model_type=moe lang_config=flores200.sampled \
  eval_on=all \
  gen_splits="[valid]" \
  --config-name generate_multi_full
```


## Access our models from HuggingFace

[Coming Soon]

## Contributors
NLLB Modeling is currently maintained by: [Vedanuj Goswami](https://github.com/vedanuj), [Shruti Bhosale](https://github.com/shruti-bh), [Anna Sun](https://github.com/annasun28), [Maha Elbayad](https://github.com/elbayadm), [Jean Maillard](https://github.com/jeanm), [James Cross](https://github.com/jhcross), [Kaushik Ram Sadagopan](https://github.com/kauterry), [Angela Fan](https://github.com/huihuifan).
## License

NLLB and fairseq(-py) is MIT-licensed available in [LICENSE](../../../LICENSE) file.
