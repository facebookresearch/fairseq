#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


def get_grid(args):
    max_update = 300000

    eff_bsz = 512

    max_sentences = 2
    update_freq = (
        1
        if args.local or "test" in args.prefix
        else int((eff_bsz * 8) / (max_sentences * args.num_nodes * args.num_gpus))
    )
    save_interval = 2000

    warmup_updates = 3000
    peak_lr = 1.5e-04

    return [
        hyperparam(
            "--train-subset",
            "train13" if args.local or "test" in args.prefix else "train",
        ),
        hyperparam(
            "--valid-subset",
            "valid"
            if args.local or "test" in args.prefix
            else "valid,valid1,valid2,valid3,valid4",
        ),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--num-workers", 2),
        hyperparam("--model-parallel-size", min(8, args.num_gpus)),
        hyperparam("--criterion", "vocab_parallel_cross_entropy"),
        hyperparam("--save-interval-updates", save_interval),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--task", "language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam("--tokens-per-sample", 1024, save_dir_key=lambda val: f"tps{val}"),
        hyperparam(
            "--arch", "transformer_lm_megatron_big", save_dir_key=lambda val: val
        ),
        hyperparam(
            "--share-decoder-input-output-embed", save_dir_key=lambda val: "share"
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--lr", peak_lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"wu{val}"
        ),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--batch-size", max_sentences, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--bucket-cap-mb", "200"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 1),
        hyperparam("--fast-stat-sync"),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
