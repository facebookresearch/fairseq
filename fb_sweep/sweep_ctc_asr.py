#!/usr/bin/env python

# Script for BART-large speech fine-tuning
# on translation task using units as input and
# a different target dictionary

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

def get_grid(args):
    grid = []

    grid += [
            hyperparam("--best-checkpoint-metric", "wer"),
            hyperparam("--keep-best-checkpoints", 1)
    ]

    #task
    grid += [
        hyperparam("--task", "audio_finetuning"),
        # hyperparam('--autoregressive'),
        # hyperparam('--no-normalize'),
        hyperparam('--labels', 'ltr'),
        hyperparam('--eval-wer')
    ]

    grid += [
        hyperparam("--ddp-backend", "legacy_ddp"),
        hyperparam("--distributed-world-size", args.num_gpus)
    ]

    grid += [
        hyperparam("--criterion", "ctc")
    ]





    # model settings
    grid += [
        # hyperparam("--arch", "wav2vec_ctc", save_dir_key=lambda val: f"arch{val}"),
        # hyperparam("--w2v-path", "/private/home/padentomasello/models/wav2vec2/wav2vec_small.pt"),
        hyperparam("--arch", "hubert_ctc", save_dir_key=lambda val: f"arch{val}"),
        hyperparam("--w2v-path", "/private/home/padentomasello/models/hubert/hubert_base_ls960.pt"),
        hyperparam("--apply-mask"),
        hyperparam("--mask-prob", 0.5),
        hyperparam("--mask-channel-length", 64),
        hyperparam("--layerdrop", 0.1),
        hyperparam("--activation-dropout", 0.1),
        hyperparam("--feature-grad-mult", 0.0),
        hyperparam("--freeze-finetune-updates", 0)
    ]

    #dataset
    grid += [
        hyperparam("--num-workers", 6),
        hyperparam("--max-tokens", 3200000, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--train-subset", "stop_train", save_dir_key=lambda val: f"trainsubset{val}"),
        hyperparam("--valid-subset", "dev_other,dev_clean,test_other,test_clean,stop_eval,stop_test"),
    ]



    grid += [
        # hyperparam("--max-source-positions", max_source_positions, save_dir_key=lambda val: f"msp{val}"),
        # hyperparam("--max-target-positions", max_target_positions, save_dir_key=lambda val: f"mtp{val}"),
        # hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        # hyperparam(
            # "--max-update", total_num_udpates, save_dir_key=lambda val: f"mu{val}"
        # ),
        # hyperparam(
            # "--freeze-finetune-num-updates", freeze_finetune_num_updates, save_dir_key=lambda val: f"mu{val}"
        # ),
        # hyperparam("--finetune-new-dictionary"),
        # hyperparam("--required-batch-size-multiple", 1),
    ]
    # # regularization
    # grid += [
        # hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        # hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        # hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        # hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
    # ]

    # # optimization settings
    grid += [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", 1e-08, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--max-update", 320000),
        hyperparam("--lr", 0.0001, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--sentence-avg")
        # hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
    ]

    # # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "tri_stage"),
        hyperparam("--phase-ratio", "[0.1,0.4,0.5]"),
        hyperparam("--final-lr-scale", 0.05)
        # hyperparam("--total-num-update", total_num_udpates),
        # hyperparam(
            # "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        # ),
    ]
    # grid += [
        # hyperparam("--lr-scheduler", "tri_stage"),
        # hyperparam("--lr", lr, save_dir_key=lambda val: f"lr{val}"),
        # hyperparam("--hold-steps", 0),
        # hyperparam("--decay-steps", 72000),
        # hyperparam("--final-lr-scale", 0.05),
        # hyperparam(
            # "--warmup-steps", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        # ),
    # ]
    # grid += [
        # hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
    # ]

    grid += [
        # hyperparam("--eval-bleu"),
        # hyperparam("--eval-bleu-args", '{"beam": 4, "max_len_b": 200, "no_repeat_ngram_size": 3}'),
        # hyperparam("--eval-bleu-detok", 'moses'),
        # hyperparam("--eval-bleu-remove-bpe"),
    ]

    # # data loading settings
    # grid += [
        # hyperparam("--num-workers", num_data_loaders),
    # ]

    # # validation and checkpoint settings
    grid += [
        # # hyperparam("--no-save"),
        hyperparam("--validate-interval", 10),
        # hyperparam("--validate-interval-updates", 1),
        # hyperparam("--validate-after-updates", freeze_finetune_num_updates),
        # hyperparam("--save-interval-updates", 5000),
        # hyperparam("--no-epoch-checkpoints"),
        # hyperparam("--reset-meters"),
        # hyperparam("--reset-optimizer"),
        # hyperparam("--reset-lr-scheduler"),
    ]

    # grid += [
        # # hyperparam("--share-all-embeddings"),
        # hyperparam("--layernorm-embedding"),
        # hyperparam("--share-decoder-input-output-embed"),
    # ]


    # # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
        hyperparam("--fp16")
    ]

    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 10),
            # hyperparam("--distributed-no-spawn"),
        ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
