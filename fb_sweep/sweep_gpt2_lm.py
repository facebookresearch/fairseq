#!/usr/bin/env python

import sweep as sweep
from sweep import hyperparam


def get_grid(args):

    max_update = 100000

    return [
        hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        # hyperparam('--memory-efficient-fp16', save_dir_key=lambda val: 'me_fp16'),
        hyperparam("--num-workers", 2),
        hyperparam("--save-interval-updates", 10000),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--task", "language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam("--tokens-per-sample", 1024, save_dir_key=lambda val: f"tps{val}"),
        # hyperparam('--arch', 'transformer_lm_gpt', save_dir_key=lambda val: val),
        hyperparam("--arch", "transformer_lm_gpt2_small", save_dir_key=lambda val: val),
        # hyperparam('--arch', 'transformer_lm_gpt2_medium', save_dir_key=lambda val: val),
        # hyperparam('--arch', 'transformer_lm_gpt2_big', save_dir_key=lambda val: val),
        hyperparam(
            "--share-decoder-input-output-embed", save_dir_key=lambda val: "share"
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", 50e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam("--warmup-updates", 10000, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--batch-size", 2, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--required-batch-size-multiple", 1),
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
