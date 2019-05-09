#!/usr/bin/env python

try:
    import sweep_chronos as sweep
    from sweep_chronos import hyperparam
except:
    import sweep
    from sweep import hyperparam

from sweep_lm_data import set_data_based_on_shortname


def get_grid(args):
    grid = []

    enable_ooo = True

    max_update = 200000
    num_data_loaders = 4
    ddp_backend = 'no_c10d'

    # 256 x 32GB Volta
    arch = 'bi_transformer_lm_huge_relu'
    max_sentences = 2
    update_freq = 8
    peak_lr = 1e-4
    #lr_scheduler = 'inverse_sqrt'
    lr_scheduler = 'polynomial'
    #grid += [hyperparam('--decoder-learned-pos', save_dir_key=lambda val: 'learnpos')]

    max_tokens = 550 * max_sentences

    set_data_based_on_shortname(args)

    # batch size
    grid += [
        hyperparam('--tokens-per-sample', 512, save_dir_key=lambda val: f'st{val}'),
        hyperparam('--max-sentences', max_sentences, save_dir_key=lambda val: f'ms{val}'),
        hyperparam('--max-tokens', max_tokens, save_dir_key=lambda val: f'mt{val}'),
        hyperparam('--update-freq', update_freq, save_dir_key=lambda val: f'uf{val}'),
    ]

    # task settings
    grid += [
        hyperparam('--task', 'odd_one_out_lm'),
    ]

    # model settings
    grid += [
        hyperparam('--arch', arch, save_dir_key=lambda val: val),
        hyperparam('--share-decoder-input-output-embed'),
    ]

    # regularization
    grid += [
        hyperparam('--dropout', 0.1, save_dir_key=lambda val: f'dr{val}'),
        hyperparam('--attention-dropout', 0.0, save_dir_key=lambda val: f'atdr{val}'),
        hyperparam('--weight-decay', 0.01, save_dir_key=lambda val: f'wd{val}'),
    ]

    # optimization settings
    grid += [
        hyperparam('--optimizer', 'adafactor', save_dir_key=lambda val: val),
        hyperparam("--decay-rate", -0.8, save_dir_key=lambda val: f"decay{val}"),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),
    ]

    # lr scheduler
    if lr_scheduler == 'inverse_sqrt':
        grid += [
            hyperparam('--lr-scheduler', 'inverse_sqrt'),
            hyperparam('--lr', peak_lr, save_dir_key=lambda val: f'lr{val}'),
            hyperparam('--warmup-init-lr', 0),
            hyperparam('--warmup-updates', 16000, save_dir_key=lambda val: f'warm{val}'),
        ]
    elif lr_scheduler == 'polynomial':
        grid += [
            hyperparam('--lr-scheduler', 'polynomial_decay'),
            hyperparam('--lr', peak_lr, save_dir_key=lambda val: f'lr{val}'),
            hyperparam('--total-num-update', max_update),
            hyperparam('--warmup-updates', 16000, save_dir_key=lambda val: f'warm{val}'),
        ]

    if enable_ooo:
        grid += [
            hyperparam('--criterion', 'odd_one_out'),
            hyperparam('--ooo-weight', [1.0], save_dir_key=lambda val: str(val)),
            hyperparam('--short-item-prob', 0.1, save_dir_key=lambda val: f'short{val}'),
        ]

    # FP16 + distributed settings
    grid += [
        hyperparam('--ddp-backend', ddp_backend),

        #hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        hyperparam('--memory-efficient-fp16', save_dir_key=lambda val: 'me_fp16'),
        hyperparam('--fp16-init-scale', 4),
        hyperparam('--threshold-loss-scale', 1),
        hyperparam('--fp16-scale-window', 128),
    ]

    # data loading settings
    grid += [
        hyperparam('--dataset-impl', 'mmap'),
        hyperparam('--num-workers', num_data_loaders),
    ]

    # validation and checkpoint settings
    grid += [
        hyperparam('--save-interval-updates', 2000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--max-update', max_update, save_dir_key=lambda val: f'mu{val}'),
    ]

    # logging settings
    grid += [
        hyperparam('--log-format', 'json'),
        hyperparam('--log-interval', 25),
    ]

    # random seed
    grid += [
        hyperparam('--seed', [1], save_dir_key=lambda val: f'seed{val}'),
    ]

    if args.local:
        grid += [
            hyperparam('--log-format', 'json'),
            hyperparam('--log-interval', 1),
        ]

    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # if config['--seq-beam'].current_value <= 8:
    #    config['--max-tokens'].current_value = 400
    # else:
    #    config['--max-tokens'].current_value = 300
    pass


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
