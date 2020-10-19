#!/usr/bin/env python

import sweep as sweep
from sweep import hyperparam


def get_grid(args):

    max_update = 500000

    return [
        hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--fast-stat-sync", save_dir_key=lambda _: "faststatsync"),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--num-workers", 2),
        hyperparam("--task", "masked_lm"),
        hyperparam("--criterion", "masked_lm"),
        hyperparam("--arch", "roberta_base", save_dir_key=lambda val: val),
        hyperparam(
            "--sample-break-mode", "complete", save_dir_key=lambda val: "cmpltdoc"
        ),
        hyperparam("--tokens-per-sample", 512, save_dir_key=lambda val: f"tps{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-6, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", 6e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--warmup-updates", 24000, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--batch-size", 32, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--update-freq", 1, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
