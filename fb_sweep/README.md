## Getting started

To train a big Transformer model on WMT'16 En-De:

```shell script
# assuming fairseq is checked out into ~/src/fairseq-py:

$ cd ~/src/fairseq-py
$ python ./fb_sweep/sweep_wmt_en2de_transformer_big.py \
  --data ~myleott/data/data-bin/wmt16_en_de_bpe32k/ \
  --prefix test_experiment_1234 \
  --num-trials -1 \
  --num-gpus 8 \
  --partition learnfair \
  --dry-run
```

Remove the `--dry-run` option to actually launch a job on the cluster.

## Configuring the sweep

Hyperparameter options can be configured in each sweep script.
For example, in `sweep_wmt_en2de_transformer_big.py` we might have:

```python
def get_grid(args):
    return [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--max-update", 300000, save_dir_key=lambda val: f"maxupd{val}"),
        # hyperparam("--save-interval", 600, save_dir_key=lambda val: f"save_interval{val}"),

        hyperparam("--arch", "transformer_vaswani_wmt_en_de_big", save_dir_key=lambda val: val),
        hyperparam("--share-all-embeddings", save_dir_key=lambda val: "shareemb"),

        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta0.9,0.98"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--warmup-init-lr", 1e-7),
        hyperparam("--warmup-updates", 4000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--lr", [1e-4, 5e-4], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),

        hyperparam("--dropout", 0.3, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--criterion", "label_smoothed_cross_entropy"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),

        hyperparam("--max-tokens", 3584, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--seed", 2, save_dir_key=lambda val: f"seed{val}"),

        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
    ]
```

Each `hyperparam(...)` corresponds to a separate command-line flag with the following structure:
```python
hyperparam(
    "--flag",  # command-line flag corresponding to this hyperparameter
    [0.01, 0.03, 0.1, 0.3, 1.0],  # a list of possible values to sweep over for this hyperparam
    save_dir_key=lambda val: f"flag{val}",  # map the hyperparam name/value to a directory name
)
```

The sweep script will use `save_dir_key` to map each hyperparam configuration to a unique directory name, typically under `/checkpoint/$USER/$DATE`.

## Other options to sweep script

- `--num-trials`: how many random hyperparam configurations should be launched (-1 means launch all possible combinations)
- `--dry-run`: see what slurm commands will be executed (in simulation mode)
- `--local`: run the command locally instead of submitting to a remote worker via slurm
- `--num-nodes`: launch job on multiple nodes


## Aggregating results into tables
Given a sweep started on e.g. 2020-12-15, we can tabulate results with

```bash
~/fairseq-py/fb_sweep/agg_results.py \
    "/checkpoint/sshleifer/2020-12-15/**/*.log" \
    --keep-cols loss,ppl,epoch,wps \
    --csv-path sweep_results.md \
    --sort-col loss
```
to save a table like
| path                    |   loss |     ppl |   epoch |    wps |
|:------------------------|-------:|--------:|--------:|-------:|
| dummylr.lr_0.0001.ngpu1 |  11.29 | 2504.39 |       1 | 7181   |
| dummylr.lr_0.0003.ngpu1 |  11.87 | 3743.49 |       1 | 7148.9 |
| dummylr.lr_3e-05.ngpu1  |  13.1  | 8811.4  |       1 | 7123.1 |

to `sweep_results.md`. To modify the table creation logic you should edit
[`agg_results.py`](./agg_results.py). Also note that passing `--csv-path xx.csv` will save a csv.

# Using Hydra
The benefits of Hydra are discussed in detail [here](../docs/hydra_integration.md).

Here is an example parameter sweep with hydra and the `submitit_slurm` launcher.
This trains language models for 50 steps once for each comma separated parameter value.

First, [install hydra-fair-plugins](https://github.com/fairinternal/hydra-fair-plugins#installation).
Then, you can run:


```bash
python fairseq_cli/hydra_train.py \
    --multirun hydra/launcher=submitit_slurm \
    hydra.launcher.gpus_per_node=1 hydra.launcher.tasks_per_node=1 \
    distributed_training.distributed_port=33333 \
    distributed_training.distributed_world_size=1 \
    hydra.launcher.partition=dev \
    optimization.max_update=50 \
    task=dummy_lm model=transformer_lm/transformer_lm_gpt \
    optimizer=adam \
    task.tokens_per_sample=512 dataset.batch_size=8 \
    optimization.lr='[0.0001],[.0003]' \
    common.log_format=json \
    common.log_interval=1  common.fp16=True \
    checkpoint.no_save_optimizer_state=True
```

+ Note that you might need to prepend `PYTHONPATH=.` to the previous command.

Since the only comma separated parameter value is `optimization.lr='[0.0001],[.0003]'`, two models will be trained.
If we specified `optimizer=adam,sgd`, four models would be trained.

Before submitting jobs to slurm, we can debug locally by changing `--multirun hydra/launcher=submitit_local`.


### Monitor Training Logs
The command above (and all multirun commands) produce a message then wait until jobs are finished.

The first line of the command will show where in your `devfair` file system we will need to look:
For example,
```
[2020-11-24 09:52:52,463][HYDRA] Submitit 'slurm' sweep output dir : multirun/2020-11-24/09-52-49
```

We can then run `tree -Ra multirun/2020-11-24/09-52-49/` to see all the files being generated, and

```bash
ls multirun/2020-11-24/09-52-49/.submitit/**/*.out | xargs tail -n 2
```
to monitor progress of each job.

To compile results into a table, you can use fb_sweep/agg_results.py, for example:

```bash
~/fairseq-py/fb_sweep/agg_results.py \
    "multirun/2020-11-24/09-52-49/**/.submitit/**/*.out" \
    --csv-path hydra_agg.md \
    -keep-cols=ppl,wps,epoch,date
```
Will produce warnings for jobs that haven't started, and make a table like below once jobs start logging

|       path |   ppl |    wps |   epoch | date                |
|-----------:|------:|-------:|--------:|:--------------------|
| 33172726_0 |  6.52 | 1440.2 |       1 | 2020-11-24-09-52-49 |
| 33172726_1 |  3.64 | 1401.8 |       1 | 2020-11-24-09-52-49 |



### Discover launcher options
```bash
python fairseq_cli/hydra_train.py hydra/launcher=submitit_slurm --cfg hydra -p hydra.launcher
```
These launcher options are similar to the command line args accepted by scripts in `fb_sweep/`.
