#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


CC25 = sorted(
    [
        "en_XX",
        "ar_AR",
        "de_DE",
        "es_XX",
        "fr_XX",
        "hi_IN",
        "it_IT",
        "ja_XX",
        "ko_KR",
        "nl_XX",
        "ru_RU",
        "zh_CN",
        "tr_TR",
        "vi_VN",
        "ro_RO",
        "my_MM",
        "ne_NP",
        "si_LK",
        "cs_CZ",
        "lt_LT",
        "kk_KZ",
        "gu_IN",
        "fi_FI",
        "et_EE",
        "lv_LV",
    ]
)

CC50 = CC25 + sorted(
    [
        "af_ZA",
        "az_AZ",
        "bn_IN",
        "fa_IR",
        "he_IL",
        "hr_HR",
        "id_ID",
        "ka_GE",
        "km_KH",
        "mk_MK",
        "ml_IN",
        "mn_MN",
        "mr_IN",
        "pl_PL",
        "ps_AF",
        "pt_XX",
        "sv_SE",
        "sw_KE",
        "ta_IN",
        "te_IN",
        "th_TH",
        "tl_XX",
        "uk_UA",
        "ur_PK",
        "xh_ZA",
    ]
)


def get_grid(args):
    grid = []

    total_num_udpates = 500000
    warmup_updates = 10000
    num_data_loaders = 4

    arch = "mbart_large"
    break_mode = "complete_doc"

    # Denoising params
    poisson_lambda = [3.5]
    mask_p = [0.3]
    mask_length = ["span-poisson"]
    replace_length = [1]
    rotate = [0]
    mask_random = [0.1]
    insert = [0]
    sentence_permute = [1.0]

    max_tokens = 1024  # 2048
    max_sentences = 32
    max_source_positions = None
    max_target_positions = None

    save_interval = 5000
    adam_eps = 1e-6
    peak_lr = 3e-4

    update_freq = 9
    seeds = [2]

    valid_subsets = "valid"

    fp16 = True

    task = "multilingual_denoising"
    criterion = "cross_entropy"

    lr_scheduler = "poly"
    weight_decay = 0.01

    continued_pretraining = True
    if continued_pretraining:
        restore_file = "/private/home/namangoyal/src/fairseq_megatron_codepush/fairseq-py/mbart.cc25/model.pt"
        grid += [hyperparam("--restore-file", restore_file)]
        grid += [
            hyperparam("--reset-lr-scheduler"),
            hyperparam("--reset-meters"),
            hyperparam("--reset-optimizer"),
            hyperparam("--reset-dataloader"),
        ]

    grid + [
        "--no-whole-word-mask-langs",
        ",".join(["ja_XX", "km_KH", "th_TH", "zh_CN", "zh_TW"]),
    ]
    if args.local:
        grid += [hyperparam("--train-subset", "valid")]

    grid += [hyperparam("--add-lang-token", save_dir_key=lambda x: "lgtkn")]
    grid += [hyperparam("--langs", ",".join(CC50), save_dir_key=lambda x: "cc50")]

    # data settings
    grid += [
        hyperparam("--dataset-impl", "mmap"),
    ]
    grid += [
        hyperparam("--bpe", "sentencepiece", save_dir_key=lambda x: "spm"),
        hyperparam(
            "--sentencepiece-model",
            "/private/home/namangoyal/src/fairseq_megatron_codepush/fairseq-py/mbart.cc25/sentence.bpe.model",
        ),
    ]
    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam("--criterion", criterion),
    ]

    grid += [
        hyperparam(
            "--multilang-sampling-alpha", 0.7, save_dir_key=lambda val: f"alp{val}"
        ),
    ]
    # Default is complete_doc
    if break_mode == "complete":
        grid += [
            hyperparam(
                "--sample-break-mode", break_mode, save_dir_key=lambda val: f"bm{val}"
            ),
        ]
    # batch size
    grid += [
        hyperparam("--tokens-per-sample", 512, save_dir_key=lambda val: f"tps{val}"),
        hyperparam("--max-tokens", max_tokens, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam(
            "--max-update", total_num_udpates, save_dir_key=lambda val: f"mu{val}"
        ),
    ]

    if max_sentences is not None:
        grid += [
            hyperparam(
                "--batch-size", max_sentences, save_dir_key=lambda val: f"ms{val}"
            ),
        ]

    if max_source_positions is not None:
        grid += [
            hyperparam(
                "--max-source-positions",
                max_source_positions,
                save_dir_key=lambda val: f"msp{val}",
            ),
        ]
    if max_target_positions is not None:
        grid += [
            hyperparam(
                "--max-target-positions",
                max_target_positions,
                save_dir_key=lambda val: f"mtp{val}",
            ),
        ]

    grid += [
        hyperparam("--encoder-normalize-before", save_dir_key=lambda val: "enb"),
        hyperparam("--decoder-normalize-before", save_dir_key=lambda val: "dnb"),
    ]

    # task settings
    grid += [
        hyperparam("--task", task),
        hyperparam("--required-batch-size-multiple", 8),
    ]

    # regularization
    grid += [
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        # hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta998"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", peak_lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", total_num_udpates),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"wrm{val}"
        ),
    ]

    if fp16:
        grid += [
            hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        ]

    # data loading settings
    grid += [
        hyperparam("--num-workers", num_data_loaders),
        hyperparam("--valid-subset", valid_subsets),
    ]

    grid += [
        hyperparam("--save-interval-updates", save_interval),
        hyperparam("--no-epoch-checkpoints"),
    ]

    grid += [
        hyperparam(
            "--poisson-lambda", poisson_lambda, save_dir_key=lambda val: f"lam{val}"
        ),
        hyperparam("--mask", mask_p, save_dir_key=lambda val: f"mask{val}"),
        hyperparam(
            "--mask-length", mask_length, save_dir_key=lambda val: f"msklen{val}"
        ),
        hyperparam(
            "--replace-length", replace_length, save_dir_key=lambda val: f"rpllen{val}"
        ),
        hyperparam("--rotate", rotate, save_dir_key=lambda val: f"rot{val}"),
        hyperparam(
            "--mask-random", mask_random, save_dir_key=lambda val: f"mskrnd{val}"
        ),
        hyperparam("--insert", insert, save_dir_key=lambda val: f"ins{val}"),
        hyperparam(
            "--permute-sentences",
            sentence_permute,
            save_dir_key=lambda val: f"prmsen{val}",
        ),
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 1000),
    ]

    # random seed
    grid += [
        hyperparam("--seed", seeds, save_dir_key=lambda val: f"seed{val}"),
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
