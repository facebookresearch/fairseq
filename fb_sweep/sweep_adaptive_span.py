#!/usr/bin/env python
"""
Usage:
    ./fb_sweep/apdative_span_sweep.py\
    -d ~daju/data/enwik8/eos-data-bin/ \
    -p enwiki8.adaptivespan \
    -t 1 -g 4 \
    --snapshot-code --snapshot-recurse-dirs fairseq,fairseq_cli,examples/truncated_bptt \
    --constraint volta32gb --partition dev
"""

import sweep
from sweep import hyperparam


def get_grid(args):
    target_batch_size = 64
    # max_batch_size_on_v100 = 16

    num_gpus = args.num_gpus * args.num_nodes
    batch_size_per_gpu = target_batch_size // num_gpus
    update_freq = target_batch_size // (batch_size_per_gpu * num_gpus)
    assert target_batch_size == update_freq * batch_size_per_gpu * num_gpus

    return [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--fp16-no-flatten-grads"),
        hyperparam("--max-update", 600000),
        hyperparam("--user-dir", "examples/adaptive_span"),
        hyperparam("--task", "truncated_bptt_lm"),
        hyperparam("--tokens-per-sample", 512),
        hyperparam("--arch", "adaptive_span", save_dir_key=lambda val: val),
        hyperparam("--n-layer", 12),
        hyperparam("--d-model", 512),
        hyperparam("--n-head", 8),
        hyperparam("--d-inner", 2048),
        hyperparam("--dropout", 0.3),
        hyperparam("--attn-span", 8192),
        hyperparam(
            "--optimizer", "adagrad_with_grad_clip", save_dir_key=lambda val: val
        ),
        hyperparam("--adagrad-clip", 0.03, save_dir_key=lambda val: f"ag_cl{val}"),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", 1000),
        hyperparam("--lr-scheduler", "fixed", save_dir_key=lambda val: val),
        hyperparam("--warmup-updates", [32000], save_dir_key=lambda val: f"wu{val}",),
        hyperparam("--batch-size-valid", batch_size_per_gpu * 2),
        hyperparam("--lr", [0.07], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--criterion", "adaptive_span_loss"),
        hyperparam("--batch-size", batch_size_per_gpu),
        hyperparam("--update-freq", update_freq),
        hyperparam("--seed", [2], save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25 if not args.local else 1),
        hyperparam(
            "--aux-loss-scaler", [0.0000005], save_dir_key=lambda val: f"loss{val}",
        ),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
