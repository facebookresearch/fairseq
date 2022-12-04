import itertools
import os
import random
import shlex
import subprocess
from collections import OrderedDict


def main(get_grid, postprocess_hyperparams, args):
    assert args.local or args.tpu is not None, "--tpu is required for TPU jobs"

    # compute all possible hyperparameter configurations
    grid = get_grid(args)
    grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    random.shuffle(grid_product)

    for i, hp_values in enumerate(grid_product):
        config = OrderedDict()
        for hp, value in zip(grid, hp_values):
            config[hp.name] = hp
            config[hp.name].current_value = value

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        # launch training
        launch_train(args, config)

        if i == args.num_trials - 1:
            break


def launch_train(args, config):
    def dry_run(msg):
        if args.dry_run:
            print(f"| dry-run: {msg}")
        return args.dry_run

    # compute save_dir
    save_dir_key = ".".join(
        filter(
            lambda save_dir_key: save_dir_key is not None,
            [hp.get_save_dir_key() for hp in config.values()],
        )
    )
    save_dir_key = save_dir_key.replace(",", "_")
    num_total_gpus = args.num_nodes * args.num_gpus
    if args.force_checkpoints_dir:
        raise NotImplementedError
    save_dir = os.path.join(
        args.checkpoints_dir,
        f"{args.prefix}.{save_dir_key}.ntpu{num_total_gpus}",
    )

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        if not dry_run(f"create directory: {save_dir}"):
            os.makedirs(save_dir)
            # os.chmod(save_dir, 0o777)

    # if has_started(save_dir) and not args.resume_checkpoints_dir:
    #    print(f'skip in progress run: {save_dir}')
    #    return

    # generate train command
    cmd_args = [
        "python",
        "/mnt/fairseq_data/fairseq-py/train.py",
        "--distributed-world-size",
        str(args.num_nodes * args.num_gpus),
        "--tpu",
    ]
    if not args.local:
        cmd_args = [
            "python",
            "-m",
            "torch_xla.distributed.xla_dist",
            "--tpu",
            args.tpu,
            "--conda-env",
            "torch-xla-nightly",
            "--",
        ] + cmd_args
    if args.data:
        cmd_args += [args.data]
    cmd_args += ["--save-dir", save_dir]
    for hp in config.values():
        if hp.name == "--fp16":
            hp.name = "--bf16"
        cmd_args.extend(map(str, hp.get_cli_args()))
    cmd_args_str = " ".join(map(shlex.quote, cmd_args))
    if args.dry_run:
        dry_run(f"train command: {cmd_args_str}")

    # initialize train log
    train_log = os.path.join(save_dir, "train.log")
    if not dry_run(f"create train.log at: {train_log}"):
        with open(train_log, "a") as train_log_h:
            train_log_h.write("")
        os.chmod(train_log, 0o777)

    if args.dry_run:
        print("| dry-run: start training")
        print(f"| dry-run: - run command: {cmd_args_str}")
    else:
        subprocess.Popen(cmd_args).wait()

    return train_log


def has_started(save_dir):
    train_log = os.path.join(save_dir, "train.log")
    if not os.path.exists(train_log):
        return False
    return True


def get_random_port():
    old_state = random.getstate()
    random.seed()
    port = random.randint(10000, 20000)
    random.setstate(old_state)
    return port
