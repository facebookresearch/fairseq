#!/usr/bin/env python
"""
Usage:

./fb_sweep/sweep_lm_enwik8_transformer_xl.py \
    -d ~daju/data/enwik8/eos-data-bin/ \
    -p enwiki8.transformer_xl \
    -t 1 -g 4 \
    --snapshot-code --snapshot-recurse-dirs fairseq,fairseq_cli,examples/truncated_bptt \
    --constraint volta32gb
"""

import sweep as sweep
from sweep import hyperparam


def get_grid(args):
    target_batch_size = 60
    max_batch_size_on_v100 = 15

    num_gpus = args.num_gpus * args.num_nodes
    batch_size_per_gpu = min(max_batch_size_on_v100, target_batch_size // num_gpus)
    update_freq = target_batch_size // (batch_size_per_gpu * num_gpus)
    assert target_batch_size == update_freq * batch_size_per_gpu * num_gpus

    return [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--max-update", 400000),
        hyperparam("--user-dir", "examples/truncated_bptt"),
        hyperparam("--task", "truncated_bptt_lm"),
        hyperparam("--tokens-per-sample", 512),
        hyperparam("--arch", "transformer_xl", save_dir_key=lambda val: val),
        hyperparam("--n-layer", 12),
        hyperparam("--d-model", 512),
        hyperparam("--n-head", 8),
        hyperparam("--d-head", 64),
        hyperparam("--d-inner", 2048),
        hyperparam("--dropout", 0.1),
        hyperparam("--dropatt", 0.0),
        hyperparam("--mem-len", 512),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--clip-norm", 0.25, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "cosine", save_dir_key=lambda val: val),
        hyperparam("--warmup-updates", 0),
        hyperparam("--lr", 0.00025, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--batch-size", batch_size_per_gpu),
        hyperparam("--update-freq", update_freq),
        hyperparam("--seed", [2], save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25 if not args.local else 1),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
