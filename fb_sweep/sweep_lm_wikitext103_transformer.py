#!/usr/bin/env python

import sweep as sweep
from sweep import hyperparam


def get_grid(args):
    return [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        #hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--max-epoch", 50),
        hyperparam("--lazy-load"),
        hyperparam("--num-workers", 4),

        hyperparam("--task", "language_modeling"),

        hyperparam("--arch", "transformer_lm_big", save_dir_key=lambda val: val),
        hyperparam("--share-decoder-input-output-embed", save_dir_key=lambda val: "shareemb"),

        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta0.9,0.98"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"initlr{val}"),
        hyperparam("--warmup-updates", 4000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--lr", 10e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--min-lr", 1e-10),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--dropout", 0.3, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--attention-dropout", [0.1], save_dir_key=lambda val: f"attndrop{val}"),
        hyperparam("--relu-dropout", [0.1], save_dir_key=lambda val: f"reludrop{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),

        # hyperparam('--update-freq', 4, save_dir_key=lambda val: f'updatefreq{val}'),
        hyperparam("--max-tokens", 1024, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--tokens-per-sample", 512, save_dir_key=lambda val: f"sampletok{val}"),

        hyperparam("--seed", [2], save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25),
    ]

def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
