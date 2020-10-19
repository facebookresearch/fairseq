#!/usr/bin/env python

import sweep as sweep
from sweep import hyperparam


def get_grid(args):

    config = "8k"  # 2k

    if config == "8k":
        max_update = 100000
        save_interval = 5000
        valid_interval = 5000
        update_freq = 1
        lr = 5.2e-4
        warmup = 5000
    else:
        max_update = 100000
        save_interval = 5000
        valid_interval = 5000
        update_freq = 4
        lr = 5e-4
        warmup = 5000

    seeds = [0]
    grid = [
        # hyperparam('--train-subset', 'train' if not args.local else 'test'),
        hyperparam("--train-subset", "valid"),
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--num-workers", 4),
        hyperparam("--task", "multilingual_masked_lm"),
        hyperparam("--criterion", "masked_lm"),
        hyperparam("--arch", "roberta_large", save_dir_key=lambda val: val),
        hyperparam("--sample-break-mode", "complete", save_dir_key=lambda val: "cmplt"),
        hyperparam("--tokens-per-sample", 512, save_dir_key=lambda val: f"tps{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-6, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 1.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--warmup-updates", warmup, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        # hyperparam('--max-tokens', 3200, save_dir_key=lambda val: f'mt{val}'),
        hyperparam("--batch-size", 12, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam(
            "--multilang-sampling-alpha", 0.7, save_dir_key=lambda val: f"s{val}"
        ),
    ]
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100),
    ]

    # random seed
    grid += [
        hyperparam("--seed", seeds, save_dir_key=lambda val: f"seed{val}"),
    ]

    grid += [
        hyperparam("--validate-interval", valid_interval),
    ]
    grid += [
        hyperparam("--save-interval-updates", save_interval),
        hyperparam("--no-epoch-checkpoints"),
    ]

    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 1),
        ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
