#!/usr/bin/env python

import sweep as sweep
from sweep import hyperparam


# These hyperparameters are tuned for RoBERTa-large
tasks = {
    "MNLI": {
        "data": "/private/home/myleott/data/data-bin/MNLI-bin",
        "--num-classes": 3,
        "--lr": 1e-5,
        "--batch-size": 32,
        "--max-update": 123873,
    },
    "QNLI": {
        "data": "/private/home/myleott/data/data-bin/QNLI-bin",
        "--num-classes": 2,
        "--lr": 1e-5,
        "--batch-size": 32,
        "--max-update": 33112,
    },
    "QQP": {
        "data": "/private/home/myleott/data/data-bin/QQP-bin",
        "--num-classes": 2,
        "--lr": 1e-5,
        "--batch-size": 32,
        "--max-update": 113272,
    },
    "RTE": {
        "data": "/private/home/myleott/data/data-bin/RTE-bin",
        "--num-classes": 2,
        "--lr": 2e-5,
        "--batch-size": 16,
        "--max-update": 2036,
    },
    "SST-2": {
        "data": "/private/home/myleott/data/data-bin/SST-2-bin",
        "--num-classes": 2,
        "--lr": 1e-5,
        "--batch-size": 32,
        "--max-update": 20935,
    },
    "MRPC": {
        "data": "/private/home/myleott/data/data-bin/MRPC-bin",
        "--num-classes": 2,
        "--lr": 1e-5,
        "--batch-size": 16,
        "--max-update": 2296,
    },
    "CoLA": {
        "data": "/private/home/myleott/data/data-bin/CoLA-bin",
        "--num-classes": 2,
        "--lr": 1e-5,
        "--batch-size": 16,
        "--max-update": 5336,
    },
    "STS-B": {
        "data": "/private/home/myleott/data/data-bin/STS-B-bin",
        "--num-classes": 1,
        "--lr": 2e-5,
        "--batch-size": 16,
        "--max-update": 3598,
        "--regression-target": True,
        "--best-checkpoint-metric": "loss",
        "--maximize-best-checkpoint-metric": False,
    },
}


# convert a dataset path to the name of the dataset
def get_save_dir_key(data_dir):
    for task_name, task_config in tasks.items():
        if task_config["data"] == data_dir:
            return task_name
    raise Exception


def get_grid(args):

    model_size = "large"

    return [
        hyperparam("--train-subset", "train" if not args.local else "valid"),
        hyperparam(
            "data",
            list(tasks.keys()),
            positional_arg=True,
            save_dir_key=lambda val: get_save_dir_key(val),
        ),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-last-checkpoints"),
        hyperparam("--no-save-optimizer-state"),
        hyperparam("--save-interval-updates", 1000),
        hyperparam("--reset-optimizer"),
        hyperparam("--reset-dataloader"),
        hyperparam("--reset-meters"),
        hyperparam("--best-checkpoint-metric", "accuracy"),
        hyperparam("--maximize-best-checkpoint-metric", [True], binary_flag=True),
        hyperparam(
            "--restore-file",
            "/private/home/myleott/roberta." + model_size + "/model.pt",
        ),
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--num-workers", 1 if not args.local else 0),
        hyperparam(
            "--task", "sentence_prediction", save_dir_key=lambda val: "sentpred"
        ),
        hyperparam("--init-token", 0, save_dir_key=lambda val: f"bos{val}"),
        hyperparam("--separator-token", 2, save_dir_key=lambda val: f"sep{val}"),
        hyperparam("--max-positions", 512),
        hyperparam("--regression-target", [False], binary_flag=True),
        hyperparam("--arch", "roberta_" + model_size, save_dir_key=lambda val: val),
        hyperparam("--bpe", "gpt2"),
        hyperparam("--criterion", "sentence_prediction"),
        hyperparam("--num-classes", [None]),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-6, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", [None], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--warmup-updates", [None], save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--total-num-update", [None]),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.01, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--batch-size", [None], save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", 1, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", [None], save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", [1], save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 25),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    # apply task-specific overrides
    t = config["data"].current_value  # current task name
    for k, v in tasks[t].items():
        assert k in config
        config[k].current_value = v

    # configure number of updates (warmup and total)
    config["--warmup-updates"].current_value = int(
        0.06 * config["--max-update"].current_value
    )
    config["--total-num-update"].current_value = config["--max-update"].current_value


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
