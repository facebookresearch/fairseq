#!/usr/bin/env python

import sweep
from sweep import hyperparam


def get_grid(args):
    target_batch_size = 128
    max_tokens = 2048
    tokens_per_sample = 512
    batch_size_per_gpu = max_tokens // tokens_per_sample
    num_gpus = args.num_gpus * args.num_nodes
    update_freq = target_batch_size // batch_size_per_gpu // num_gpus

    return [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--max-update", 50000),
        hyperparam("--task", "language_modeling"),
        #hyperparam("--arch", "hf_gpt2", save_dir_key=lambda val: val),
        hyperparam("--arch", "transformer_lm_gpt", save_dir_key=lambda val: val),
        hyperparam("--share-decoder-input-output-embed", save_dir_key=lambda val: "shareemb"),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam(
            "--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta0.9,0.98"
        ),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--lr", 5e-4, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--warmup-updates", 4000, save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"initlr{val}"),
        hyperparam(
            "--tokens-per-sample", tokens_per_sample, save_dir_key=lambda val: f"sampletok{val}"
        ),
        hyperparam(
            "--sample-break-mode", "none", save_dir_key=lambda val: f"break{val}"
        ),
        hyperparam("--max-tokens", max_tokens, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"updatefreq{val}"),
        hyperparam("--seed", [2], save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
