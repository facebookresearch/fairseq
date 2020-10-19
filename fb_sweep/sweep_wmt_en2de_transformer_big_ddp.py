#!/usr/bin/env python

import sweep
from sweep import hyperparam
from sweep_wmt_en2de_transformer_big_common import get_common_grid


COMMON_GRID = get_common_grid()


def get_grid(args):
    return COMMON_GRID + [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--distributed-wrapper", "DDP", save_dir_key=lambda val: f"{val}"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
