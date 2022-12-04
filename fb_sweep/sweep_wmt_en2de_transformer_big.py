#!/usr/bin/env python

import sweep
from sweep import hyperparam


def get_grid(args):
    """
    Replicates the `16-bit+cumul+2x lr` results from Table 1 of
    "Scaling Neural Machine Translation" (https://arxiv.org/abs/1806.00187)
    """
    return [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--ddp-backend", "fully_sharded", save_dir_key=lambda val: val),
        # hyperparam("--cpu-offload"),
        # hyperparam("--no-reshard-after-forward"),
        hyperparam("--max-epoch", 70),
        # equivalent to training on 16x GPUs
        hyperparam(
            "--update-freq",
            16 if not args.local else 1,
            save_dir_key=lambda val: f"updatefreq{val}",
        ),
        hyperparam("--arch", "transformer_wmt_en_de_big", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            [True],
            binary_flag=True,
            save_dir_key=lambda val: "shareemb",
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam(
            "--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta0.9,0.98"
        ),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"initlr{val}"),
        hyperparam("--warmup-updates", 4000, save_dir_key=lambda val: f"warmup{val}"),
        # use double the default learning rate, since we're using --update-freq=16
        hyperparam("--lr", 10e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", 1e-9),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--dropout", 0.3, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--criterion", "label_smoothed_cross_entropy"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
        hyperparam("--max-tokens", 3584, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--seed", [2], save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100 if not args.local else 10),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
