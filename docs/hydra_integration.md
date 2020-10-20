

## Hydra

Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.

## Train models with hydra interface

#### Provide parameters in `.yaml` files
For example, if we'd like to train a language model with transformer, we could provide parameters in yaml files. Note that the modules used (task, model, criterion, optimizer, lr scheduler) in training must be migrated with hydra interface already (See session below).

- Provide top level choices on which generic parameter file, and which modules to use: `config/config.yaml`, this will look like for example:

```
defaults:
  - task: language_modeling
  - model: transformer_lm
  - criterion: cross_entropy
  - optimizer: adam
  - lr_scheduler: inverse_sqrt
```

- Provide generic parameters common across different jobs: `config.yaml`
- Provide task parameters: `config/task/language_modeling.yaml`
- Provide model parameters: `config/model/transformer_lm.yaml`
- Provide criterion parameters: `config/criterion/cross_entropy.yaml`
- Provide optimizer parameters: `config/optimizer/adam.yaml`
- Provide lr_scheduler parameters `config/lr_scheduler/inverse_sqrt.yaml`

#### Command line overriding
`train_hydra.py` is the main entry point for training with hydra interface. If we specify all parameters we want in `.yaml` files, then we could simply use command:

```
# task.data is requested field marked by `???` in yaml
python fairseq_cli/train_hydra.py \
task.data=/private/home/abaevski/data/wiki103 \
```

Alternatively, if we need to override certain params from the command line, we could do so as below (note the structure of where each parameter sits)

```
python fairseq_cli/train_hydra.py
task=language_modeling \
task.data=/private/home/abaevski/data/wiki103 \
task.tokens_per_sample=512 \
task.sample_break_mode=none \
model=transformer_lm \
model.share_decoder_input_output_embed=true \
model.dropout=0.1 \
optimizer=adam \
optimizer.adam_betas="'(0.9, 0.98)'" \
optimizer.weight_decay=0.01 \
lr_scheduler=inverse_sqrt \
lr_scheduler.warmup_updates=4000 \
lr_scheduler.warmup_init_lr=1e-07 \
criterion=cross_entropy \
common.fp16=true \
common.log_format=json \
common.log_interval=1 \
dataset.max_tokens=1024 \
dataset.num_workers=4 \
optimization.update_freq=[16] \
optimization.max_update=50000 \
optimization.clip_norm=0.0 \
optimization.lr=[0.0005] \
checkpoint.save_dir=/checkpoint/mtian/transformer_wikitext-103-hydra-args-cli \
checkpoint.save_interval_updates=10
```

## Migrate existing/Creating new modules to hydra interface

In each of the modules we want to migrated/create with hydra interface, fundamentally we need to

- Provide a dataclass that layouts the parameters used in the module.

- Modify the builder and/or constructor that previously takes `argparse.Namespace` argument `args`, into taking `omegaconf.DictConfig` config objects. At this moment we allow `Union[omegaconf.DictConfig, argparse.Namespace]` to support compatibility.

- For `add_args()`, we need to extract argument from the dataclass defined in the same file, and append them into `parser`. This is also to support compatibility. This is simply supported with `gen_parser_from_dataclass` API, see examples files below.

#### Migrated examples:

- Task: `fairseq/tasks/language_modeling.py`

- Model: `fairseq/models/transformer_lm.py`

- Criterion: `fairseq/criterions/adaptive_loss.py` and `fairseq/criterions/cross_entropy.py`

- Optimizer: `fairseq/optim/adam.py` and `fairseq/optim/nag.py`

- LR scheduler: `fairseq/optim/lr_scheduler/cosine_lr_scheduler.py` and `fairseq/optim/lr_scheduler/inverse_square_root_schedule.py`


## Interpolate parameters across different places

## Support of legacy interface
If you still like to pass legacy style arguments in command line, `fairseq_cli/train.py` can support this. Internally it coverted `args` into hydra config objects whenever there are migrated modules aligned.

```
python fairseq_cli/train.py --task language_modeling \
/private/home/abaevski/data/wiki103 \
--save-dir /checkpoint/mtian/transformer_wikitext-103-hydra-args-cli \
--arch transformer_lm --share-decoder-input-output-embed \
--dropout 0.1 \
--optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
--tokens-per-sample 512 --sample-break-mode none \
--max-tokens 1024 --update-freq 16 \
--fp16 \
--max-update 50000 --log-format json --log-interval 1 --num-workers 4 \
--save-interval-updates 10
```
