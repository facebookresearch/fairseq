# Fully Sharded Data Parallel (FSDP)

## Overview
Recent work by [Microsoft](https://arxiv.org/abs/1910.02054) and
[Google](https://arxiv.org/abs/2004.13336) has shown that data parallel
training can be made significantly more efficient by sharding the model
parameters and optimizer state across data parallel workers. These ideas are
encapsulated in the new **`FullyShardedDataParallel` (FSDP)** wrapper provided
by [fairscale](https://github.com/facebookresearch/fairscale/).

Compared to PyTorch DDP:
* FSDP produces identical results as PyTorch DDP (it's still synchronous data parallel training)
* FSDP shards parameters (FP16 + FP32) and optimizer state across data parallel GPUs
* FSDP is faster than PyTorch DDP because the optimizer step is sharded, and the communication can be overlapped with the forward pass
* FSDP enables training 13B parameter models on 8 GPUs and 175B parameter models on 128 GPUs

FSDP is fully supported in fairseq via the following new arguments:
* `--ddp-backend=fully_sharded`: enables full sharding via FSDP
* `--cpu-offload`: offloads the optimizer state and FP32 model copy to CPU (combine with `--optimizer=cpu_adam`)
* `--no-reshard-after-forward`: increases training speed for large models (1B+ params) and is similar to ZeRO stage 2
* other popular options (`--fp16`, `--update-freq`, `--checkpoint-activations`, `--offload-activations`, etc.) continue to work as normal

<details><summary>Limitations</summary><p>

FSDP currently has several limitations compared to fairseq's default DDP backend (PyTorch DDP):
* while FSDP is full compatible with pointwise Optimizers (e.g., Adam, AdamW, Adadelta, Adamax, SGD, etc.), it is not currently compatible with non-pointwise Optimizers (e.g., Adagrad, Adafactor, LAMB, etc.)
* FSDP depends on flattening the parameters, so models that currently require `--fp16-no-flatten-grads` may not be supported

See the [fairscale docs](https://fairscale.readthedocs.io/en/latest/api/nn/fsdp_tips.html) for a more detailed
explanation of these and other limitations.

</p></details>

<details><summary>How it works</summary><p>

<img width="800" alt="Fully Sharded Data Parallel" src="https://user-images.githubusercontent.com/231798/110406775-c2de0000-8050-11eb-9718-fbfc4510a76a.png">

See the [fairscale docs](https://fairscale.readthedocs.io/en/latest/api/nn/fsdp_tips.html) for a more detailed
explanation of how FSDP works.

</p></details>

## Example usage

The following examples illustrate how to train a very large language model with
13 billion parameters on 1 GPU by offloading parameters and optimizer states to
CPU, or on 8 GPUs by fully sharding the params and optimizer states across GPUs.

These examples use the WikiText-103 dataset for demonstration purposes, but
in practice a much larger dataset will be needed to achieve good results.
Follow the [instructions here](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md#1-preprocess-the-data)
to preprocess the WikiText-103 dataset using the GPT-2/RoBERTa vocabulary.

### 13B params on 1 V100 GPU (with CPU offloading)

The following command trains a 13B parameter GPT-3 model on a single V100 GPU
using the `--cpu-offload` feature to offload parameters and optimizer states to
CPU. In this setting, the optimizer step (Adam) happens on CPU. We also use the
`--checkpoint-activations` feature (sometimes called [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html)),
which further saves memory in exchange for a small increase in computation.

**Requirements:**
- Install the latest master version of fairscale: `pip install git+https://github.com/facebookresearch/fairscale.git@master`
- You'll need 32GB of GPU memory and ~256GB of system memory to train the 13B param model.
- If you have less system memory, the 6.7B param model can be trained with ~128GB of system memory, just set `--arch transformer_lm_gpt3_6_7`
- We use the CPU Adam optimizer from [DeepSpeed](https://github.com/microsoft/DeepSpeed), so you'll need to `pip install deepspeed` before running the command.

**Notes:**
- The command will take ~5 minutes to start training, during which time it will appear to be hung, since randomly initializing 13B weights can be slow.
- The `--cpu-offload` feature requires training in mixed precision (`--fp16`).
- Tune the `OMP_NUM_THREADS` env variable for best performance with CPU offloading.
- The example command below stops training after 10 steps (`--max-update 10`) and does not save checkpoints (`--no-save`).

```bash
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0 \
    fairseq-train data-bin/wikitext-103-roberta-bpe-bin \
    --ddp-backend fully_sharded --fp16 --fp16-init-scale 4 \
    --cpu-offload --checkpoint-activations \
    --task language_modeling --tokens-per-sample 2048 --batch-size 8 \
    --arch transformer_lm_gpt3_13 \
    --optimizer cpu_adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler polynomial_decay --warmup-updates 5 --total-num-update 10 \
    --max-update 10 --no-save --log-format json --log-interval 1
```

<details><summary>Example output</summary><p>

```
(...)
2021-03-08 12:29:51 | INFO | fairseq_cli.train | num. model params: 13,110,865,920 (num. trained: 13,110,865,920)
(...)
2021-03-08 12:29:51 | INFO | fairseq_cli.train | training on 1 devices (GPUs/TPUs)
2021-03-08 12:29:51 | INFO | fairseq_cli.train | max tokens per GPU = None and batch size per GPU = 8
(...)
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000100, betas=(0.900000, 0.980000), weight_decay=0.000000, adam_w=1
(...)
2021-03-08 12:31:36 | INFO | train_inner | {"epoch": 1, "update": 0.0, "loss": "16.475", "ppl": "91120.8", "wps": "0", "ups": "0", "wpb": "16384", "bsz": "8", "num_updates": "1", "lr": "2e-05", "gnorm": "20.751", "loss_scale": "4", "train_wall": "99", "gb_free": "9.3", "wall": "105"}
2021-03-08 12:32:33 | INFO | train_inner | {"epoch": 1, "update": 0.0, "loss": "16.446", "ppl": "89281.6", "wps": "288.7", "ups": "0.02", "wpb": "16384", "bsz": "8", "num_updates": "2", "lr": "4e-05", "gnorm": "19.777", "loss_scale": "4", "train_wall": "57", "gb_free": "9.3", "wall": "161"}
2021-03-08 12:33:12 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 2.0
2021-03-08 12:33:51 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 1.0
2021-03-08 12:34:45 | INFO | train_inner | {"epoch": 1, "update": 0.001, "loss": "25.22", "ppl": "3.90691e+07", "wps": "123.4", "ups": "0.01", "wpb": "16384", "bsz": "8", "num_updates": "3", "lr": "6e-05", "gnorm": "131.281", "loss_scale": "1", "train_wall": "133", "gb_free": "9.3", "wall": "294"}
2021-03-08 12:35:43 | INFO | train_inner | {"epoch": 1, "update": 0.001, "loss": "18.079", "ppl": "276809", "wps": "285.5", "ups": "0.02", "wpb": "16384", "bsz": "8", "num_updates": "4", "lr": "8e-05", "gnorm": "13.776", "loss_scale": "1", "train_wall": "57", "gb_free": "9.3", "wall": "351"}
2021-03-08 12:36:35 | INFO | train_inner | {"epoch": 1, "update": 0.001, "loss": "23.729", "ppl": "1.39088e+07", "wps": "316.7", "ups": "0.02", "wpb": "16384", "bsz": "8", "num_updates": "5", "lr": "0.0001", "gnorm": "72.774", "loss_scale": "1", "train_wall": "52", "gb_free": "9.3", "wall": "403"}
2021-03-08 12:37:28 | INFO | train_inner | {"epoch": 1, "update": 0.001, "loss": "20.429", "ppl": "1.41203e+06", "wps": "307.6", "ups": "0.02", "wpb": "16384", "bsz": "8", "num_updates": "6", "lr": "8e-05", "gnorm": "60.846", "loss_scale": "1", "train_wall": "53", "gb_free": "9.3", "wall": "456"}
2021-03-08 12:38:27 | INFO | train_inner | {"epoch": 1, "update": 0.001, "loss": "18.965", "ppl": "511684", "wps": "279.4", "ups": "0.02", "wpb": "16384", "bsz": "8", "num_updates": "7", "lr": "6e-05", "gnorm": "22.687", "loss_scale": "1", "train_wall": "59", "gb_free": "9.3", "wall": "515"}
2021-03-08 12:39:18 | INFO | train_inner | {"epoch": 1, "update": 0.001, "loss": "18.345", "ppl": "332887", "wps": "319.1", "ups": "0.02", "wpb": "16384", "bsz": "8", "num_updates": "8", "lr": "4e-05", "gnorm": "8.451", "loss_scale": "1", "train_wall": "51", "gb_free": "9.3", "wall": "566"}
2021-03-08 12:40:11 | INFO | train_inner | {"epoch": 1, "update": 0.002, "loss": "18.262", "ppl": "314336", "wps": "305.9", "ups": "0.02", "wpb": "16384", "bsz": "8", "num_updates": "9", "lr": "2e-05", "gnorm": "6.457", "loss_scale": "1", "train_wall": "54", "gb_free": "9.3", "wall": "620"}
2021-03-08 12:41:04 | INFO | train_inner | {"epoch": 1, "update": 0.002, "loss": "17.556", "ppl": "192686", "wps": "311.8", "ups": "0.02", "wpb": "16384", "bsz": "8", "num_updates": "10", "lr": "0", "gnorm": "5.796", "loss_scale": "1", "train_wall": "53", "gb_free": "9.3", "wall": "673"}
2021-03-08 12:41:04 | INFO | fairseq_cli.train | Stopping training due to num_updates: 10 >= max_update: 10
2021-03-08 12:41:04 | INFO | fairseq_cli.train | begin validation on "valid" subset
2021-03-08 12:43:15 | INFO | valid | {"epoch": 1, "valid_loss": "17.953", "valid_ppl": "253807", "valid_wps": "1868.4", "valid_wpb": "15400.2", "valid_bsz": "7.6", "valid_num_updates": "10"}
2021-03-08 12:43:15 | INFO | fairseq_cli.train | end of epoch 1 (average epoch stats below)
2021-03-08 12:43:15 | INFO | train | {"epoch": 1, "train_loss": "19.351", "train_ppl": "668509", "train_wps": "210.9", "train_ups": "0.01", "train_wpb": "16384", "train_bsz": "8", "train_num_updates": "10", "train_lr": "0", "train_gnorm": "36.26", "train_loss_scale": "1", "train_train_wall": "667", "train_gb_free": "9.3", "train_wall": "804"}
2021-03-08 12:43:15 | INFO | fairseq_cli.train | done training in 798.6 seconds
```

</p></details>

### 13B params on 8 V100 GPUs (with full parameter + optimizer state sharding)

FSDP can also shard the parameters and optimizer states across multiple GPUs,
reducing memory requirements significantly. On 8 x 32GB GPUs, sharding enables
training the same 13B parameter model *without offloading the parameters to
CPU*. However, without CPU offloading we'd only be able to fit a batch size of
1 per GPU, which would cause training speed to suffer.

We obtain the best performance on 8 GPUs by combining full sharding and CPU
offloading. The following command trains the same 13B parameter GPT-3 model as
before on 8 x 32GB V100 GPUs; training speed increases superlinearly from ~310
words per second to ~3200 words per second.

```bash
OMP_NUM_THREADS=20 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    fairseq-train data-bin/wikitext-103-roberta-bpe-bin \
    --ddp-backend fully_sharded --fp16 --fp16-init-scale 4 \
    --cpu-offload --checkpoint-activations \
    --task language_modeling --tokens-per-sample 2048 --batch-size 8 \
    --arch transformer_lm_gpt3_13 \
    --optimizer cpu_adam --adam-betas "(0.9,0.98)" \
    --lr 0.0001 --lr-scheduler polynomial_decay --warmup-updates 5 --total-num-update 10 \
    --max-update 10 --no-save --log-format json --log-interval 1
```

<details><summary>Example output</summary><p>

```
(...)
2021-03-08 18:04:09 | INFO | fairseq_cli.train | num. model params: 13,110,865,920 (num. trained: 13,110,865,920)
(...)
2021-03-08 18:04:09 | INFO | fairseq_cli.train | training on 8 devices (GPUs/TPUs)
2021-03-08 18:04:09 | INFO | fairseq_cli.train | max tokens per GPU = None and batch size per GPU = 8
(...)
Adam Optimizer #0 is created with AVX2 arithmetic capability.
Config: alpha=0.000100, betas=(0.900000, 0.980000), weight_decay=0.000000, adam_w=1
(...)
2021-03-08 18:05:06 | INFO | train_inner | {"epoch": 1, "update": 0.001, "loss": "16.408", "ppl": "86945.6", "wps": "0", "ups": "0", "wpb": "131072", "bsz": "64", "num_updates": "1", "lr": "2e-05", "gnorm": "18.27", "loss_scale": "4", "train_wall": "47", "gb_free": "9.3", "wall": "56"}
2021-03-08 18:05:45 | INFO | train_inner | {"epoch": 1, "update": 0.002, "loss": "16.352", "ppl": "83644.3", "wps": "3283.4", "ups": "0.03", "wpb": "131072", "bsz": "64", "num_updates": "2", "lr": "4e-05", "gnorm": "18.411", "loss_scale": "4", "train_wall": "40", "gb_free": "9.3", "wall": "96"}
2021-03-08 18:06:21 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 2.0
2021-03-08 18:06:56 | INFO | fairseq.trainer | NOTE: gradient overflow detected, ignoring gradient, setting loss scale to: 1.0
2021-03-08 18:07:37 | INFO | train_inner | {"epoch": 1, "update": 0.006, "loss": "23.682", "ppl": "1.34537e+07", "wps": "1176.6", "ups": "0.01", "wpb": "131072", "bsz": "64", "num_updates": "3", "lr": "6e-05", "gnorm": "119.682", "loss_scale": "1", "train_wall": "111", "gb_free": "9.3", "wall": "208"}
2021-03-08 18:08:18 | INFO | train_inner | {"epoch": 1, "update": 0.007, "loss": "18.988", "ppl": "519921", "wps": "3189.1", "ups": "0.02", "wpb": "131072", "bsz": "64", "num_updates": "4", "lr": "8e-05", "gnorm": "14.934", "loss_scale": "1", "train_wall": "41", "gb_free": "9.3", "wall": "249"}
2021-03-08 18:08:59 | INFO | train_inner | {"epoch": 1, "update": 0.008, "loss": "20.08", "ppl": "1.10798e+06", "wps": "3223.1", "ups": "0.02", "wpb": "131072", "bsz": "64", "num_updates": "5", "lr": "0.0001", "gnorm": "59.92", "loss_scale": "1", "train_wall": "41", "gb_free": "9.3", "wall": "289"}
2021-03-08 18:09:39 | INFO | train_inner | {"epoch": 1, "update": 0.009, "loss": "18.323", "ppl": "327980", "wps": "3256.6", "ups": "0.02", "wpb": "131072", "bsz": "64", "num_updates": "6", "lr": "8e-05", "gnorm": "37.425", "loss_scale": "1", "train_wall": "40", "gb_free": "9.3", "wall": "330"}
2021-03-08 18:10:20 | INFO | train_inner | {"epoch": 1, "update": 0.01, "loss": "17.264", "ppl": "157354", "wps": "3188.7", "ups": "0.02", "wpb": "131072", "bsz": "64", "num_updates": "7", "lr": "6e-05", "gnorm": "10.824", "loss_scale": "1", "train_wall": "41", "gb_free": "9.3", "wall": "371"}
2021-03-08 18:11:01 | INFO | train_inner | {"epoch": 1, "update": 0.011, "loss": "16.794", "ppl": "113647", "wps": "3230", "ups": "0.02", "wpb": "131072", "bsz": "64", "num_updates": "8", "lr": "4e-05", "gnorm": "5.616", "loss_scale": "1", "train_wall": "41", "gb_free": "9.3", "wall": "411"}
2021-03-08 18:11:39 | INFO | train_inner | {"epoch": 1, "update": 0.012, "loss": "16.706", "ppl": "106938", "wps": "3384", "ups": "0.03", "wpb": "131072", "bsz": "64", "num_updates": "9", "lr": "2e-05", "gnorm": "5.318", "loss_scale": "1", "train_wall": "39", "gb_free": "9.3", "wall": "450"}
2021-03-08 18:12:19 | INFO | train_inner | {"epoch": 1, "update": 0.013, "loss": "16.548", "ppl": "95796.2", "wps": "3274.4", "ups": "0.02", "wpb": "131072", "bsz": "64", "num_updates": "10", "lr": "0", "gnorm": "5.22", "loss_scale": "1", "train_wall": "40", "gb_free": "9.3", "wall": "490"}
2021-03-08 18:12:19 | INFO | fairseq_cli.train | Stopping training due to num_updates: 10 >= max_update: 10
2021-03-08 18:12:19 | INFO | fairseq_cli.train | begin validation on "valid" subset
2021-03-08 18:12:45 | INFO | valid | {"epoch": 1, "valid_loss": "16.624", "valid_ppl": "101000", "valid_wps": "10855.9", "valid_wpb": "123202", "valid_bsz": "60.5", "valid_num_updates": "10"}
2021-03-08 18:12:45 | INFO | fairseq_cli.train | end of epoch 1 (average epoch stats below)
2021-03-08 18:12:45 | INFO | train | {"epoch": 1, "train_loss": "18.114", "train_ppl": "283776", "train_wps": "2567.8", "train_ups": "0.02", "train_wpb": "131072", "train_bsz": "64", "train_num_updates": "10", "train_lr": "0", "train_gnorm": "29.562", "train_loss_scale": "1", "train_train_wall": "480", "train_gb_free": "9.3", "train_wall": "516"}
2021-03-08 18:12:45 | INFO | fairseq_cli.train | done training in 509.9 seconds
```

</p></details>
