#!/usr/bin/env python
import sweep
from sweep import hyperparam


MODEL = {
    "levenshtein_transformer": "lev_base",
    "levenshtein_transformer_wmt_en_de": "lev_base",
    "levenshtein_transformer_big": "lev_big",
    "levenshtein_transformer_wmt_en_de_big": "lev_big",
    "nonautoregressive_transformer": "nat",
    "nacrf_transformer": "nat_crf",
    "iterative_nonautoregressive_transformer": "inat",
    "cmlm_transformer": "cmlm",
    "insertion_transformer": "ins",
}


def get_at_grid(args):
    """
    Auto-regressive Transformer
    """
    return [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam("--max-update", 300000),
        # equivalent to training on 16x GPUs
        # hyperparam('--update-freq', 16, save_dir_key=lambda val: f'updatefreq{val}'),
        hyperparam("--arch", ["transformer_small"], save_dir_key=lambda val: val),
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
        #  use double the default learning rate, since we're using --update-freq=16
        hyperparam("--lr", 0.0005, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", 1e-9),
        hyperparam("--clip-norm", 25, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--weight-decay", 0.0001, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--criterion", "label_smoothed_cross_entropy"),
        hyperparam("--label-smoothing", 0.1, save_dir_key=lambda val: f"ls{val}"),
        hyperparam("--max-tokens", 4096, save_dir_key=lambda val: f"maxtok{val}"),
        hyperparam("--seed", [2], save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--keep-last-epochs", 15),
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 100),
    ]


def get_grid_levenshtein(args):
    return [
        # task, model, criterion
        hyperparam("--task", "translation_lev"),
        hyperparam(
            "--arch",
            "levenshtein_transformer_wmt_en_de",
            save_dir_key=lambda val: MODEL[val],
        ),
        # hyperparam('--arch', [
        #     'levenshtein_transformer_wmt_en_de_big',
        #     'levenshtein_transformer_wmt_en_de'
        # ],
        #            save_dir_key=lambda val: MODEL[val]),
        hyperparam("--criterion", "label_smoothed_dual_imitation"),
        hyperparam("--noise", "random_delete"),
        # task specific
        hyperparam("--fixed-validation-seed", 7),
        hyperparam("--append-bos", binary_flag=True),
        # model
        hyperparam("--encoder-learned-pos", binary_flag=True),
        hyperparam(
            "--decoder-learned-pos",
            binary_flag=True,
            save_dir_key=lambda val: f"lp" if val else f"sp",
        ),
        hyperparam("--share-all-embeddings", binary_flag=True),
        hyperparam(
            "--apply-bert-init",
            binary_flag=True,
            save_dir_key=lambda val: f"bert" if val else f"",
        ),
        hyperparam("--early-exit", "(6,6,6)", save_dir_key=lambda val: f"ext-{val}"),
        # general
        hyperparam("--activation-fn", "gelu", save_dir_key=lambda val: f"act-{val}"),
        # hyperparam('--max-tokens', 8192, save_dir_key=lambda val: f'b{val}'),
        hyperparam("--max-tokens", 4096, save_dir_key=lambda val: f"b8192"),
        hyperparam("--update-freq", 2),
        hyperparam("--fp16", binary_flag=True),
        hyperparam("--optimizer", "adam"),
        hyperparam("--lr", 0.0005, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", "1e-09"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--max-update", 400000),
        hyperparam("--warmup-updates", 10000),
        hyperparam("--keep-last-epochs", 15),
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--warmup-init-lr", "1e-07"),
        hyperparam("--adam-betas", "(0.9, 0.999)"),
        hyperparam("--dropout", 0.3),
        hyperparam("--label-smoothing", 0.1),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--save-interval-updates", 10000),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 5),
        hyperparam("--seed", 2),
        # hyperparam('--seed', [1, 11], save_dir_key=lambda val: f'prefix{val % 10}'),
        # hyperparam('--seed', [3, 5, 7, 13, 15, 17], save_dir_key=lambda val: f'prefix{val % 10}'),
        # hyperparam('--seed', 5, save_dir_key=lambda val: f'fuse-0.{val}'),
    ]


def get_grid_progressive(args):
    return [
        # task, model, criterion
        hyperparam("--task", "translation_lev"),
        hyperparam("--arch", "progressive_transformer"),
        hyperparam("--criterion", "label_smoothed_dual_imitation"),
        hyperparam("--noise", "full_mask"),
        # task specific
        hyperparam("--fixed-validation-seed", 7),
        hyperparam("--append-bos", binary_flag=True),
        # model
        hyperparam("--encoder-learned-pos", binary_flag=True),
        hyperparam(
            "--decoder-learned-pos",
            binary_flag=True,
            save_dir_key=lambda val: f"lp" if val else f"sp",
        ),
        hyperparam("--share-all-embeddings", binary_flag=True),
        hyperparam(
            "--apply-bert-init",
            binary_flag=True,
            save_dir_key=lambda val: f"bert" if val else f"",
        ),
        # model specific
        hyperparam("--passing-unk", binary_flag=True, save_dir_key=lambda val: f"pu"),
        hyperparam("--pred-length-offset", binary_flag=True),
        # hyperparam('--sg-length-pred', binary_flag=True, save_dir_key=lambda val: f'sg' if val else f''),
        hyperparam("--output-checker", binary_flag=True),
        hyperparam("--pred-length-format", "mean"),
        hyperparam("--length-loss-factor", 0.1, save_dir_key=lambda val: f"lf{val}"),
        hyperparam("--fixed-depth", 9, save_dir_key=lambda val: f"d-{val}"),
        # general
        hyperparam("--activation-fn", "gelu", save_dir_key=lambda val: f"act-{val}"),
        hyperparam("--max-tokens", 5192, save_dir_key=lambda val: f"bz{val}"),
        hyperparam("--fp16", binary_flag=True),
        hyperparam("--optimizer", "adam"),
        hyperparam("--lr", 0.0005, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", "1e-09"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--max-update", 400000),
        hyperparam("--warmup-updates", 10000),
        hyperparam("--keep-last-epochs", 15),
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--warmup-init-lr", "1e-07"),
        hyperparam("--adam-betas", "(0.9, 0.999)"),
        hyperparam("--dropout", 0.3),
        hyperparam("--label-smoothing", 0.1),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--save-interval-updates", 10000),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 5),
    ]


def get_grid_nat(args):
    return [
        # task, model, criterion
        hyperparam("--task", "translation_lev"),
        hyperparam(
            "--arch",
            "nonautoregressive_transformer",
            save_dir_key=lambda val: MODEL[val],
        ),
        hyperparam("--criterion", "label_smoothed_dual_imitation"),
        # task specific
        hyperparam("--fixed-validation-seed", 7),
        hyperparam("--append-bos", binary_flag=True),
        hyperparam("--noise", "full_mask"),
        # model
        hyperparam("--encoder-learned-pos", binary_flag=True),
        hyperparam(
            "--decoder-learned-pos",
            binary_flag=True,
            save_dir_key=lambda val: f"lp" if val else f"sp",
        ),
        hyperparam("--share-all-embeddings", binary_flag=True),
        hyperparam(
            "--apply-bert-init",
            binary_flag=True,
            save_dir_key=lambda val: f"bert" if val else f"",
        ),
        # length prediction
        hyperparam("--pred-length-offset", binary_flag=True),
        # hyperparam('--sg-length-pred', binary_flag=True, save_dir_key=lambda val: f'sg' if val else f''),
        hyperparam("--length-loss-factor", 0.1, save_dir_key=lambda val: f"lf{val}"),
        hyperparam(
            "--src-embedding-copy", binary_flag=True, save_dir_key=lambda val: "cp"
        ),
        # n-gram loss
        # hyperparam('--ngram-predictor',
        #            4,
        #            save_dir_key=lambda val: f'{val}-gram'),
        # general
        hyperparam("--activation-fn", "gelu", save_dir_key=lambda val: f"act-{val}"),
        hyperparam("--max-tokens", 4096, save_dir_key=lambda val: f"b{val}"),
        hyperparam("--fp16", binary_flag=True),
        hyperparam("--optimizer", "adam"),
        hyperparam("--lr", 0.0005, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", "1e-09"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--max-update", 400000),
        hyperparam("--warmup-updates", 10000),
        hyperparam("--keep-last-epochs", 15),
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--warmup-init-lr", "1e-07"),
        hyperparam("--adam-betas", "(0.9, 0.999)"),
        hyperparam("--dropout", 0.3),
        hyperparam("--label-smoothing", 0.1),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--save-interval-updates", 10000),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 5),
        # hyperparam('--seed', [1, 2, 3, 4, 5, 6, 7], save_dir_key=lambda val: f'rb-{val}'),
    ]


def get_grid_nacrf(args):
    return [
        # task, model, criterion
        hyperparam("--task", "translation_lev"),
        hyperparam("--arch", "nacrf_transformer", save_dir_key=lambda val: MODEL[val]),
        hyperparam("--criterion", "nat_loss"),
        # task specific
        hyperparam("--fixed-validation-seed", 7),
        hyperparam("--noise", "full_mask"),
        # model
        hyperparam("--encoder-learned-pos", binary_flag=True),
        hyperparam(
            "--decoder-learned-pos",
            binary_flag=True,
            save_dir_key=lambda val: f"lp" if val else f"sp",
        ),
        hyperparam("--share-all-embeddings", binary_flag=True),
        hyperparam(
            "--apply-bert-init",
            binary_flag=True,
            save_dir_key=lambda val: f"bert" if val else f"",
        ),
        # length prediction
        hyperparam("--pred-length-offset", binary_flag=True),
        # hyperparam('--sg-length-pred', binary_flag=True, save_dir_key=lambda val: f'sg' if val else f''),
        hyperparam("--length-loss-factor", 0.1, save_dir_key=lambda val: f"lf{val}"),
        # general
        hyperparam("--activation-fn", "gelu", save_dir_key=lambda val: f"act-{val}"),
        hyperparam("--max-tokens", 8192, save_dir_key=lambda val: f"b{val}"),
        hyperparam("--fp16", binary_flag=True),
        hyperparam("--optimizer", "adam"),
        hyperparam("--lr", 0.0005, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", "1e-09"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--max-update", 400000),
        hyperparam("--warmup-updates", 10000),
        hyperparam("--keep-last-epochs", 15),
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--warmup-init-lr", "1e-07"),
        hyperparam("--adam-betas", "(0.9, 0.999)"),
        hyperparam("--dropout", 0.3),
        hyperparam("--label-smoothing", 0.1),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--save-interval-updates", 10000),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 5),
    ]


def get_grid_inat(args):
    return [
        # task, model, criterion
        hyperparam("--task", "translation_lev"),
        hyperparam(
            "--arch",
            "iterative_nonautoregressive_transformer",
            save_dir_key=lambda val: MODEL[val],
        ),
        hyperparam("--criterion", "label_smoothed_dual_imitation"),
        # task specific
        hyperparam("--fixed-validation-seed", 7),
        hyperparam("--append-bos", binary_flag=True),
        hyperparam("--noise", "full_mask"),
        # model
        hyperparam("--encoder-learned-pos", True, binary_flag=True),
        hyperparam(
            "--decoder-learned-pos",
            True,
            binary_flag=True,
            save_dir_key=lambda val: f"lp" if val else f"sp",
        ),
        hyperparam("--share-all-embeddings", binary_flag=True),
        hyperparam(
            "--apply-bert-init",
            binary_flag=True,
            save_dir_key=lambda val: f"bert" if val else f"",
        ),
        # iterative refinement settings
        hyperparam("--train-step", 3, save_dir_key=lambda val: f"iter{val}"),
        hyperparam("--dae-ratio", 0.5, save_dir_key=lambda val: f"dae{val}"),
        hyperparam(
            "--stochastic-approx", True, binary_flag=True, save_dir_key=lambda val: "sa"
        ),
        # length prediction
        hyperparam("--pred-length-offset", binary_flag=True),
        # hyperparam('--sg-length-pred', binary_flag=True, save_dir_key=lambda val: f'sg' if val else f''),
        hyperparam("--length-loss-factor", 0.1, save_dir_key=lambda val: f"lf{val}"),
        # hyperparam('--src-embedding-copy', [True, False],
        #            binary_flag=True,
        #            save_dir_key=lambda val: 'copy'),
        # n-gram loss
        # hyperparam('--ngram-predictor',
        #            4,
        #            save_dir_key=lambda val: f'{val}-gram'),
        # general
        hyperparam("--activation-fn", "gelu", save_dir_key=lambda val: f"{val}"),
        hyperparam("--max-tokens", 2048, save_dir_key=lambda val: f"b{val}"),
        hyperparam("--update-freq", 2, save_dir_key=lambda val: f"u{val}"),
        hyperparam("--fp16", binary_flag=True),
        hyperparam("--optimizer", "adam"),
        hyperparam("--lr", 0.0005, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", "1e-09"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--max-update", 400000),
        hyperparam("--warmup-updates", 10000),
        hyperparam("--keep-last-epochs", 5),
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--warmup-init-lr", "1e-07"),
        hyperparam("--adam-betas", "(0.9, 0.999)"),
        hyperparam("--dropout", 0.3),
        hyperparam("--label-smoothing", 0.1),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--save-interval-updates", 10000),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 5),
        # hyperparam('--seed', [1, 2, 3, 4, 5, 6, 7], save_dir_key=lambda val: f'rb-{val}'),
    ]


def get_grid_cmlm(args):
    return [
        # task, model, criterion
        hyperparam("--task", "translation_lev"),
        hyperparam("--arch", "cmlm_transformer", save_dir_key=lambda val: MODEL[val]),
        hyperparam("--criterion", "label_smoothed_dual_imitation"),
        # task specific
        hyperparam("--fixed-validation-seed", 7),
        hyperparam("--append-bos", binary_flag=True),
        hyperparam("--noise", "random_mask"),
        # model
        hyperparam("--encoder-learned-pos", True, binary_flag=True),
        hyperparam(
            "--decoder-learned-pos",
            True,
            binary_flag=True,
            save_dir_key=lambda val: f"lp" if val else f"sp",
        ),
        hyperparam("--share-all-embeddings", binary_flag=True),
        hyperparam(
            "--apply-bert-init",
            binary_flag=True,
            save_dir_key=lambda val: f"bert" if val else f"",
        ),
        # length prediction
        hyperparam("--pred-length-offset", binary_flag=True),
        # hyperparam('--sg-length-pred', binary_flag=True, save_dir_key=lambda val: f'sg' if val else f''),
        hyperparam("--length-loss-factor", 0.1, save_dir_key=lambda val: f"lf{val}"),
        # general
        hyperparam("--activation-fn", "gelu", save_dir_key=lambda val: f"{val}"),
        hyperparam("--max-tokens", 4096, save_dir_key=lambda val: f"b{val}"),
        hyperparam("--update-freq", 2, save_dir_key=lambda val: f"u{val}"),
        hyperparam("--fp16", binary_flag=True),
        hyperparam("--optimizer", "adam"),
        hyperparam("--lr", 0.0005, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", "1e-09"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--max-update", 400000),
        hyperparam("--warmup-updates", 10000),
        hyperparam("--keep-last-epochs", 5),
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--warmup-init-lr", "1e-07"),
        hyperparam("--adam-betas", "(0.9, 0.999)"),
        hyperparam("--dropout", 0.3),
        hyperparam("--label-smoothing", 0.1),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--save-interval-updates", 10000),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 5),
    ]


def get_grid_insertion(args):
    return [
        # task, model, criterion
        hyperparam("--task", "translation_lev"),
        hyperparam(
            "--arch", "insertion_transformer", save_dir_key=lambda val: MODEL[val]
        ),
        hyperparam("--criterion", "label_smoothed_dual_imitation"),
        hyperparam("--noise", "random_delete"),
        # task specific
        hyperparam("--fixed-validation-seed", 7),
        hyperparam("--append-bos", binary_flag=True),
        # model
        hyperparam("--encoder-learned-pos", binary_flag=True),
        hyperparam(
            "--decoder-learned-pos",
            binary_flag=True,
            save_dir_key=lambda val: f"lp" if val else f"sp",
        ),
        hyperparam("--share-all-embeddings", binary_flag=True),
        hyperparam(
            "--apply-bert-init",
            binary_flag=True,
            save_dir_key=lambda val: f"bert" if val else f"",
        ),
        hyperparam(
            "--label-tau",
            1,
            save_dir_key=lambda val: f"tau{val}" if val < 1000 else f"uniform",
        ),
        # general
        hyperparam("--activation-fn", "gelu", save_dir_key=lambda val: f"act-{val}"),
        # hyperparam('--max-tokens', 6144, save_dir_key=lambda val: f'bz{val}'),
        hyperparam("--max-tokens", 8192, save_dir_key=lambda val: f"b{val}"),
        hyperparam("--fp16", binary_flag=True),
        hyperparam("--optimizer", "adam"),
        hyperparam("--lr", 0.0005, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", "1e-09"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--max-update", 400000),
        hyperparam("--warmup-updates", 10000),
        hyperparam("--warmup-init-lr", "1e-07"),
        hyperparam("--adam-betas", "(0.9, 0.999)"),
        hyperparam("--dropout", 0.3),
        hyperparam("--label-smoothing", 0.1),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--save-interval-updates", 10000),
        hyperparam("--keep-last-epochs", 15),
        hyperparam("--keep-interval-updates", 5),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 5),
        # hyperparam('--seed', [1, 2, 3, 4, 5, 6, 7], save_dir_key=lambda val: f'rb-{val}'),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300
    # decoder_embed_dim = config['--decoder_embed_dim'].current_value
    #
    # config['--decoder-ffn-embed-dim'] = 4 * decoder_embed_dim
    # config['--decoder-attention-heads'] = decoder_embed_dim // 16

    # dataset, name = sweep_datasets(config['--seed'].current_value)
    # args.data = dataset
    # args.prefix = name
    # args.seed = 1


if __name__ == "__main__":
    sweep.main(get_grid_nacrf, postprocess_hyperparams)
    # sweep.main(get_grid_levenshtein_pp, postprocess_hyperparams)
    # sweep.main(get_at_grid, postprocess_hyperparams)
    # sweep.main(get_grid_inat, postprocess_hyperparams)
    # sweep.main(get_grid_nat, postprocess_hyperparams)
    # sweep.main(get_grid_levenshtein, postprocess_hyperparams)
    # sweep.main(get_grid_progressive, postprocess_hyperparams)
    # sweep.main(get_grid_cmlm, postprocess_hyperparams)
    # sweep.main(get_grid_insertion, postprocess_hyperparams)
    # sweep.main(get_grid_reinforce, postprocess_hyperparams)
