#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sweep
from sweep import hyperparam


PREDIFINED_GRID_FUNCTION = {}


def register_grid(name):
    def register_grid_func(fn):
        if name not in PREDIFINED_GRID_FUNCTION:
            PREDIFINED_GRID_FUNCTION[name] = fn
        return fn

    return register_grid_func


def get_predefined_grid(name):
    if name not in PREDIFINED_GRID_FUNCTION:
        return []
    else:
        return PREDIFINED_GRID_FUNCTION[name]()


def add_extra_options_func(parser):
    parser.add_argument("--max-update", help="max update", default=40000)
    parser.add_argument(
        "--finetune-from-model",
        help="finetune from a pretrained model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-tokens", help="max tokens per batch", type=int, default=3584
    )
    parser.add_argument("--arch", default="transformer")
    parser.add_argument("--task", default="translation")

    # equivalent to training on 16x GPUs
    parser.add_argument("--update-freq", default=4)
    # use double the default learning rate, since we're using --update-freq=16
    # per token learning should be approximately constant;
    # ideally momentent and 2nd momentent of adam should be adjusted accordingly but less important
    parser.add_argument("--lr", default=10e-4)
    parser.add_argument(
        "--ddp-backend",
        default=None,
    )
    parser.add_argument(
        "--source-lang", help="source language for translation", type=str
    )
    parser.add_argument(
        "--target-lang", help="target language for translation", type=str
    )


@register_grid("transformer_16_16")
def get_transformer_16_16_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 16, save_dir_key=lambda val: f"ELS{val}"),
        hyperparam("--decoder-layers", 16, save_dir_key=lambda val: f"DLS{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"ATTDRP{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"RELDRP{val}"),
    ]


@register_grid("mbart_large")
def get_transformer_mbart_large_grid():
    return [
        hyperparam("--arch", "mbart_large", save_dir_key=lambda val: val),
        hyperparam("--lang-tok-style", "mbart"),
        hyperparam(
            "--layernorm-embedding", binary_flag=True, save_dir_key=lambda val: "lnemb"
        ),
        hyperparam("--encoder-learned-pos"),
        hyperparam("--decoder-learned-pos"),
        hyperparam("--encoder-normalize-before"),
        hyperparam("--decoder-normalize-before"),
        hyperparam("--share-all-embeddings"),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"ATTDRP{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"RELDRP{val}"),
    ]


@register_grid("transformer_12_12")
def get_transformer_12_12_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"ELS{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"DLS{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"ATTDRP{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"RELDRP{val}"),
    ]


def get_grid(args):
    max_update = args.max_update
    task = args.task

    grids = [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--max-update", max_update),
        hyperparam(
            "--source-lang", args.source_lang, save_dir_key=lambda val: f"@{val}"
        ),
        hyperparam(
            "--target-lang", args.target_lang, save_dir_key=lambda val: f"@{val}"
        ),
        hyperparam(
            "--update-freq", args.update_freq, save_dir_key=lambda val: f"up{val}"
        ),
        hyperparam("--task", task),
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
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", 1e-9),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--dropout", 0.3, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--criterion", "label_smoothed_cross_entropy"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
        hyperparam("--seed", [2], save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100 if not args.local else 10),
    ]

    if args.ddp_backend:
        grids.append(
            hyperparam(
                "--ddp-backend", args.ddp_backend, save_dir_key=lambda val: f"{val}"
            )
        )

    if args.max_tokens:
        grids.append(
            hyperparam(
                "--max-tokens", args.max_tokens, save_dir_key=lambda val: f"maxtok{val}"
            ),
        )
    if args.finetune_from_model:
        grids.append(hyperparam("--finetune-from-model", args.finetune_from_model))

    arch_grid = get_predefined_grid(args.arch)
    arch_grid = (
        arch_grid
        if arch_grid
        else [
            hyperparam("--arch", args.arch, save_dir_key=lambda val: val),
        ]
    )

    grids += arch_grid

    return grids


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
