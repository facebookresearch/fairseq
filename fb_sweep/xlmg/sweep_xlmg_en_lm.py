#!/usr/bin/env python
"""
Example usage:

    PYTHONPATH=. ./fb_sweep/xlmg/sweep_xlmg_en_lm.py \
            --num-trials 1 --num-gpus 8 --num-nodes 1 \
            --model-size 125M_xlmg_h2_2021 \
            --prefix xlmg.125m \
            --partition learnaccel

This sweep script takes some additional optional arguments. See add_extra_options_func
for more details.
"""

import os
from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

SHARDED_ENGLISH_DATA = None
if os.path.exists("/fsx/sshleifer/data/rc-bin-40"):  # AWS
    SHARDED_ENGLISH_DATA = ":".join(
        (f"/fsx/sshleifer/data/rc-bin-40/shard{i}" for i in range(40))
    )
    UNSHARDED_EN_DATA = (
        "/fsx/myleott/data/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/"
    )
elif os.path.exists("/data/xlmg"):  # Azure (H2 2021)
    UNSHARDED_EN_DATA = (
        "/data/xlmg/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin"
    )
elif os.path.exists("/nfs2/"):  # Azure (H1 2021)
    UNSHARDED_EN_DATA = (
        "/mnt/data/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/"
    )
else:  # FAIR Cluster
    SHARDED_ENGLISH_DATA = ":".join(
        (f"/checkpoint/sshleifer/data-bin/rc-bin-40/shard{i}" for i in range(40))
    )
    UNSHARDED_EN_DATA = "/private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin"
    if not os.path.exists(f"{UNSHARDED_EN_DATA}/dict.txt"):  # Mini-RSC hack
        UNSHARDED_EN_DATA = "/large_experiments/xlmg/models/sshleifer/data/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin"
# This contains the roberta training data + the cc100_en training data split into 40 random shards.
# It is the same data as /private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin, but sharded


def add_extra_options_func(parser):
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="use synthetic data and only train for 50 steps (for benchmarking)",
    )
    parser.add_argument(
        "--model-size", help="model configuration, see get_grid for available options"
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="tokens_per_sample")
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument(
        "--debug-train-on-small-subset",
        action="store_true",
        help="only load a single shard of data from one datasource (OpenWebText), "
        "which reduces startup time and is useful for debugging",
    )
    parser.add_argument(
        "--optimizer",
        "--opt",
        default="adam",
        choices=["adam", "adam8bit", "cpu_adam"],
        help="which optimizer to use",
    )
    parser.add_argument("--scale-attn", action="store_true", default=False)
    parser.add_argument("--scale-fc", action="store_true", default=False)
    parser.add_argument("--scale-heads", "--sh", action="store_true")
    parser.add_argument("--lr", default=None, type=float, help="overrides default lr")
    parser.add_argument("--no-fp16-adam", action="store_true", default=False)
    parser.add_argument(
        "--bs", default=None, type=int, help="overrides default local batch size"
    )
    parser.add_argument(
        "--no-ckpt",
        default=False,
        action="store_true",
        help="dont checkpoint activations",
    )
    parser.add_argument("--scale-resids", default=False, action="store_true")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--save-interval", default=1000, type=int)
    parser.add_argument("--dropout", default=None)
    parser.add_argument("--end-learning-rate", default=None, type=float)
    parser.add_argument(
        "--zero2",
        action="store_true",
        help="use ZeRO-2 instead of ZeRO-3, which speeds up training by ~5% at the "
        "cost of more memory usage; ideal for dense equiv. models <10B params",
    )


def get_base_model_config(layers, model_dim, heads):
    return [
        hyperparam("--arch", "transformer_lm_gpt", save_dir_key=lambda val: val),
        hyperparam("--activation-fn", "gelu"),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam("--decoder-layers", layers, save_dir_key=lambda val: f"nlay{val}"),
        hyperparam(
            "--decoder-embed-dim", model_dim, save_dir_key=lambda val: f"emb{val}"
        ),
        hyperparam("--decoder-ffn-embed-dim", 4 * model_dim),
        hyperparam("--decoder-attention-heads", heads),
    ]


def add_adam8bit_config_(optimizer_config):
    optimizer_config.extend(
        [
            hyperparam("--use-sharded-state"),
            hyperparam("--stable-emb"),
            hyperparam("--no-scale-embedding"),
            hyperparam("--block-wise"),
        ]
    )


def add_cpu_adam_config_(optimizer_config):
    optimizer_config.extend(
        [
            hyperparam("--optimizer", "cpu_adam"),
            hyperparam("--cpu-offload", save_dir_key=lambda _: "cpuoff"),
            hyperparam("--offload-activations", save_dir_key=lambda _: "offloadact"),
        ]
    )


def get_grid(args):
    num_gpus = args.num_gpus * args.num_nodes
    training_tokens = int(300e9)  # matches GPT-3

    # Set this to 0 on AWS to avoid segfaults
    num_dataloading_workers = 2 if not os.path.exists("/fsx") else 0

    if args.debug_train_on_small_subset:
        train_subset = "train13"
        assert args.prefix.startswith(
            "test"
        ), "please ensure that --prefix starts with 'test' when using --debug-train-on-small-subset"
    else:
        train_subset = "train"

    # TODO the original dense training runs in H1 2021 used a single validation
    # set coming from CC-News. If you need to have comparable valid_ppl to those
    # runs, then set this to False. Otherwise True is preferred, since it will
    # aggregate all of the valid sets for CC-News, Books, Wikipedia, etc.
    combine_valid_sets = True

    if args.data is None:
        args.data = UNSHARDED_EN_DATA
    assert os.path.exists(args.data), f"Could not find data path: {args.data}"

    if os.path.exists("/nfs2/") or os.path.exists("/data/xlmg") or args.local:
        args.snapshot_code = False  # containers don't support snapshot_code
    else:
        args.snapshot_code = True

    # Model configuration based on size
    M = 1024 * 1024
    warmup_tokens = int(375 * M)
    weight_decay = 0.01
    if args.model_size == "125M_xlmg_h2_2021":
        assert num_gpus >= 8
        model_config = get_base_model_config(layers=12, model_dim=768, heads=12)
        batch_size_tokens = int(0.5 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 6e-4
        dropout = 0.1
    elif args.model_size == "355M_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/dense/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms1.uf4.mu572204.s1.ngpu64
        assert num_gpus >= 8
        model_config = get_base_model_config(layers=24, model_dim=1024, heads=16)
        batch_size_tokens = int(0.5 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 3e-4
        dropout = 0.0
    elif args.model_size == "1.3B_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/dense/1.3B/few_shot.roberta+cc100.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256
        assert num_gpus >= 32
        model_config = get_base_model_config(layers=24, model_dim=2048, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 8
        learning_rate = 2e-4
        dropout = 0.1
    elif args.model_size == "2.7B_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/dense/2.7B/gpt3_2.7B.layers32.emb2560.head32.cpt.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu357.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu286102.s1.ngpu128
        assert num_gpus >= 128
        model_config = get_base_model_config(layers=32, model_dim=2560, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 4
        learning_rate = 1.6e-4
        dropout = 0.1
    elif args.model_size == "6.7B_xlmg_h2_2021":
        assert num_gpus >= 64
        model_config = get_base_model_config(layers=32, model_dim=4096, heads=32)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.2e-4
        dropout = 0.1
    elif args.model_size == "13B_xlmg_h2_2021":
        assert num_gpus >= 64
        model_config = get_base_model_config(layers=40, model_dim=5120, heads=40)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.0e-4

        dropout = 0.1
    elif args.model_size == "175B_xlmg_h2_2021":
        # assert num_gpus >= 512
        model_config = get_base_model_config(layers=96, model_dim=12288, heads=96)
        # model_config += [
        #    hyperparam("--offload-activations", save_dir_key=lambda _: "offloadact"),
        # ]
        batch_size_tokens = int(4.0 * M)
        max_batch_size_per_gpu = 4
        learning_rate = 0.6e-4
        dropout = 0.1

    elif args.model_size == "kitchen_sink":
        assert num_gpus >= 64
        assert args.no_fp16_adam
        assert args.seq_len == 2048
        # 2.7B config
        model_config = get_base_model_config(layers=32, model_dim=2560, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.6e-4
        dropout = 0.1
        model_config.extend(
            [
                hyperparam(
                    "--activation-fn", "relu_squared", save_dir_key=lambda val: val
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown --model-size argument: {args.model_size}")

    if args.dropout is not None:
        dropout = dropout

    # Batch size logic
    batch_size_seqs = batch_size_tokens // args.seq_len
    if args.bs is not None:
        max_batch_size_per_gpu = args.bs
    batch_size_per_gpu = min(max_batch_size_per_gpu, batch_size_seqs // num_gpus)
    update_freq = batch_size_seqs // (batch_size_per_gpu * num_gpus)
    assert (
        batch_size_tokens == update_freq * batch_size_per_gpu * num_gpus * args.seq_len
    )

    max_update = training_tokens // batch_size_tokens
    warmup_updates = warmup_tokens // batch_size_tokens

    log_interval = 10 if not args.local else 1

    if args.benchmark:
        # Overrides for speed benchmarking
        args.data = None
        task_config = [
            hyperparam("--task", "dummy_lm", save_dir_key=lambda val: val),
            hyperparam(
                "--tokens-per-sample",
                args.seq_len,
                save_dir_key=lambda val: f"tps{val}",
            ),
            hyperparam("--dict-size", 51200 - 4),
            hyperparam("--disable-validation"),
        ]
        max_update = 50
        warmup_updates = 50
        log_interval = 5
    else:
        task_config = [
            hyperparam("--task", "language_modeling"),
            hyperparam(
                "--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"
            ),
            hyperparam(
                "--tokens-per-sample",
                args.seq_len,
                save_dir_key=lambda val: f"tps{val}",
            ),
        ]

    # Optimizer config
    optimizer = args.optimizer
    optimizer_config = [
        hyperparam("--optimizer", optimizer, save_dir_key=lambda val: val)
    ]
    if not args.no_fp16_adam and optimizer != "adam8bit":
        optimizer_config.append(
            hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam")
        )
    if optimizer == "adam":
        pass  # defaults set elsewhere
    elif optimizer == "adam8bit":
        add_adam8bit_config_(model_config)
    elif optimizer == "cpu_adam":
        optimizer_config.extend(
            [hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam")]
        )
        add_cpu_adam_config_(model_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    grid = []

    def H(*args, **kwargs):
        """Add a hyperparameter"""
        grid.append(hyperparam(*args, **kwargs))

    # if args.stable:
    #     H("--stable-emb", True, binary_flag=True, save_dir_key=lambda x: 'stable_emb')
    #     H("--no-scale-embedding")

    if args.restore_file:
        grid += [
            hyperparam("--restore-file", args.restore_file),
        ]
    if combine_valid_sets:
        grid += [hyperparam("--combine-val")]
    else:
        grid += [hyperparam("--ignore-unused-valid-subsets")]
    grid += [
        hyperparam("--train-subset", train_subset),
        hyperparam("--num-workers", num_dataloading_workers),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", args.save_interval),
        hyperparam(
            "--no-epoch-checkpoints"
        ),  # only save checkpoints based on num steps
        # hyperparam("--keep-interval-updates", 1),  # only keep the most recent checkpoint
        # hyperparam("--no-save-optimizer-state-on-training-finished"),
        # hyperparam("--save-async"),
        hyperparam("--ddp-backend", "fully_sharded", save_dir_key=lambda val: "fsdp"),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fp16-init-scale", 4),
    ]

    if not args.no_ckpt:
        H("--checkpoint-activations")

    if args.zero2:
        grid += [
            hyperparam("--no-reshard-after-forward", save_dir_key=lambda val: "zero2")
        ]
    grid += model_config
    grid += task_config
    grid += optimizer_config

    lr_to_use = learning_rate if args.lr is None else args.lr
    grid += [
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam(
            "--adam-betas",
            "(0.9, 0.98)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1]),
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam("--adam-eps", args.eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", lr_to_use, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"wu{val}"
        ),
        hyperparam("--dropout", dropout, save_dir_key=lambda val: f"dr{val}"),
        hyperparam(
            "--attention-dropout", dropout, save_dir_key=lambda val: f"atdr{val}"
        ),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
        hyperparam(
            "--batch-size", batch_size_per_gpu, save_dir_key=lambda val: f"ms{val}"
        ),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", log_interval),
    ]

    if args.end_learning_rate is not None:
        H(
            "--end-learning-rate",
            args.end_learning_rate,
            save_dir_key=lambda x: f"end_lr_{x}",
        )
    H(
        "--scale-attn",
        args.scale_attn,
        binary_flag=True,
        save_dir_key=lambda x: "ln_attn" if x else "",
    )
    H(
        "--scale-fc",
        args.scale_fc,
        binary_flag=True,
        save_dir_key=lambda x: "ln_fc" if x else "",
    )
    H(
        "--scale-heads",
        args.scale_heads,
        binary_flag=True,
        save_dir_key=lambda x: "scale_heads" if x else "",
    )
    H(
        "--scale-resids",
        args.scale_resids,
        binary_flag=True,
        save_dir_key=lambda x: "scale_resids" if x else "",
    )
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # Check hyperparam value in config.keys(), to avoid mis-specifying in get_grid.
    if "--clip-norm-type" in config.keys():
        norm_type = config["--clip-norm-type"].current_value
        assert norm_type in [
            "l2",
            "inf",
        ], f"Invalid --clip-norm-type of {norm_type}! Only 'l2' and 'inf' supported!"


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
