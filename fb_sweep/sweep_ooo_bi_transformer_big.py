#!/usr/bin/env python

try:
    import sweep_chronos as sweep
    from sweep_chronos import hyperparam
except:
    import sweep
    from sweep import hyperparam

from sweep_lm_data import set_data_based_on_shortname


def get_grid(args):
    grid = [
        hyperparam('--curriculum', 200),
    ]

    enable_ooo = False
    enable_finetune = True
    num_classes = 3

    max_update = 260000
    num_data_loaders = 4
    ddp_backend = 'no_c10d'

    # 64 x 16GB Volta
    arch = 'bi_transformer_lm_big'
    max_sentences = 16
    update_freq = 2
    peak_lr = 2.5e-4
    lr_scheduler = 'inverse_sqrt'
    #lr_scheduler = 'polynomial'
    #grid += [hyperparam('--decoder-learned-pos', save_dir_key=lambda val: 'learnpos')]
    dropout = 0.1

    optimizer = 'adam'
    #optimizer = 'lamb'

    max_tokens = 550 * max_sentences
    save_interval_updates = 2000

    set_data_based_on_shortname(args)

    if enable_finetune:
        lr_scheduler = 'fixed'
        peak_lr = [2e-5]
        max_sentences = [32]
        #peak_lr = 2e-5
        #max_sentences = 32
        update_freq = 1
        max_update = 300000
        dropout = [0.1]
        save_interval_updates = 250
        grid += [
            hyperparam('--unmask-curr-state', [True], binary_flag=True, save_dir_key=lambda val: 'unmask'),
            hyperparam('--final-dropout', [0.0, 0.1], save_dir_key=lambda val: f'fdrop{val}'),
            hyperparam('--extra-layer', ['tanh', 'relu'], save_dir_key=lambda val: f'extra{val}'),
            hyperparam('--init-token', 2),
            hyperparam('--reset-meters'),
            hyperparam('--valid-subset', 'valid,test'),

            # this will cause validation every 250 updates, but will not save checkpoints
            hyperparam('--no-save'),
        ]

    # batch size
    if not enable_finetune:
        grid += [
            hyperparam('--tokens-per-sample', 512, save_dir_key=lambda val: f'st{val}'),
        ]
    grid += [
        hyperparam('--max-sentences', max_sentences, save_dir_key=lambda val: f'ms{val}'),
        hyperparam('--max-tokens', max_tokens, save_dir_key=lambda val: f'mt{val}'),
        hyperparam('--update-freq', update_freq, save_dir_key=lambda val: f'uf{val}'),
    ]

    # model settings
    grid += [
        hyperparam('--arch', arch, save_dir_key=lambda val: val),
        hyperparam('--activation-fn', 'gelu_accurate', save_dir_key=lambda val: val),
        hyperparam('--share-decoder-input-output-embed'),
    ]

    # regularization
    grid += [
        hyperparam('--dropout', dropout, save_dir_key=lambda val: f'dr{val}'),
        hyperparam('--attention-dropout', 0.0, save_dir_key=lambda val: f'atdr{val}'),
        hyperparam('--weight-decay', 0.01, save_dir_key=lambda val: f'wd{val}'),
    ]

    # optimization settings
    if optimizer == 'adam':
        grid += [
            hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
            hyperparam('--adam-betas', '(0.9, 0.999)', save_dir_key=lambda val: 'beta2_999'),
        ]
    elif optimizer == 'lamb':
        grid += [
            hyperparam('--optimizer', 'lamb', save_dir_key=lambda val: val),
            hyperparam('--lamb-betas', '(0.9, 0.999)', save_dir_key=lambda val: 'beta2_999'),
        ]
    grid += [
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
    elif lr_scheduler == 'fixed':
        grid += [
            hyperparam('--lr-scheduler', 'fixed'),
            hyperparam('--lr', peak_lr, save_dir_key=lambda val: f'lr{val}'),
        ]

    assert enable_ooo ^ enable_finetune
    if enable_ooo:
        grid += [
            hyperparam('--task', 'odd_one_out_lm'),
            hyperparam('--criterion', 'odd_one_out'),
            hyperparam('--ooo-weight', [1.0], save_dir_key=lambda val: str(val)),
            hyperparam('--short-item-prob', 0.1, save_dir_key=lambda val: f'short{val}'),
        ]
    elif enable_finetune:
        grid += [
            hyperparam('--task', 'sentence_classification'),
            hyperparam('--criterion', 'sentence_classification'),
            hyperparam('--reset-optimizer'),
            hyperparam('--num-classes', num_classes),
        ]

    # FP16 + distributed settings
    grid += [
        hyperparam('--ddp-backend', ddp_backend),
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        #hyperparam('--memory-efficient-fp16', save_dir_key=lambda val: 'me_fp16'),
        hyperparam('--fp16-init-scale', 4),
        hyperparam('--threshold-loss-scale', 1),
        hyperparam('--fp16-scale-window', 128),
    ]

    # data loading settings
    grid += [
        hyperparam('--dataset-impl', 'mmap' if not enable_finetune else 'cached'),
        hyperparam('--num-workers', num_data_loaders),
    ]

    # validation and checkpoint settings
    grid += [
        hyperparam('--save-interval-updates', save_interval_updates),
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
