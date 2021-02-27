#!/usr/bin/env python

import sweep as sweep
from sweep import hyperparam


def get_grid(args):
    return [
        hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam("--fp16"),
        # hyperparam('--memory-efficient-fp16', save_dir_key=lambda val: 'me_fp16'),
        hyperparam("--num-workers", 2),
        hyperparam("--log-interval", 1),
        hyperparam("--optimizer", "adam"),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-save"),
        hyperparam("--task", "dummy_lm", save_dir_key=lambda val: val),
        hyperparam("--tokens-per-sample", 512),
        hyperparam("--max-sentences", 2),
        # hyperparam('--arch', 'transformer_lm_gpt', save_dir_key=lambda val: val),
        hyperparam("--arch", "transformer_lm_gpt2_tiny"),
        hyperparam("--log-format", "json"),
        hyperparam("--max-update", 10),
        hyperparam("--lr", 3e-4),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
