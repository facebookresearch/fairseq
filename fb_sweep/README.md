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
