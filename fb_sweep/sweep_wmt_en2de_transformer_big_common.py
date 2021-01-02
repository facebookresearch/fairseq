#!/usr/bin/env python

from sweep import hyperparam


def get_common_grid():
    return [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--ddp-backend", "c10d", save_dir_key=lambda val: "no_c10d"),
        hyperparam("--max-epoch", 70),
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
        hyperparam("--lr", 0.001, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", 1e-9),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--dropout", 0.3, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--criterion", "label_smoothed_cross_entropy"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
        hyperparam("--max-tokens", 3584, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--seed", [2], save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25),
        hyperparam("--save-interval", 1),
    ]
