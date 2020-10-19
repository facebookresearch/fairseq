import itertools
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict

import libfb.py.fbpkg as fbpkg


def main(get_grid, postprocess_hyperparams, args):
    assert args.hostgroup is not None, "--hostgroup is required for Chronos jobs"

    # compute all possible hyperparameter configurations
    grid = get_grid(args)
    grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    random.shuffle(grid_product)

    # build train fbpkg
    if not args.local and args.fbpkg is not None:
        train_fbpkg = args.fbpkg
    else:
        # build train.par
        if args.debug:
            mode = "dbg"
        elif args.local:
            mode = "dev-nosan"
        else:
            mode = "opt"
        buck_cmd = [
            "/usr/local/bin/buck",
            "build",
            "@mode/" + mode,
            "deeplearning/projects/fairseq-py:fb_train",
        ]
        buck_cmd_str = " ".join(map(shlex.quote, buck_cmd))
        if args.dry_run:
            print(f"| dry-run: {buck_cmd_str}")
        else:
            subprocess.Popen(
                buck_cmd,
                cwd=os.path.join(
                    "/data/users",
                    os.environ["USER"],
                    "fbsource/fbcode",
                ),
            ).wait()

        if args.dry_run:
            print(f"| dry_run: build fbpkg")
        elif args.local:
            train_fbpkg = None
        else:
            train_fbpkg = fbpkg.build_version(
                "fairseq",
                build_config=fbpkg.BuildConfig(
                    paths=[
                        os.path.join(
                            "/data/users",
                            os.environ["USER"],
                            "fbsource/fbcode",
                            "buck-out/gen/deeplearning/projects/fairseq-py/fb_train.par",
                        )
                    ],
                ),
                ephemeral=True,
                expire="2w",
            )[0].identifier

        if args.build_only:
            sys.exit(0)

    if args.dry_run:
        train_fbpkg = "fb_train.par"

    for i, hp_values in enumerate(grid_product):
        config = OrderedDict()
        for hp, value in zip(grid, hp_values):
            config[hp.name] = hp
            config[hp.name].current_value = value

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        # launch training
        launch_train(args, config, train_fbpkg)

        if i == args.num_trials - 1:
            break


def launch_train(args, config, train_fbpkg):
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
    x = int(time.time())
    if not args.force_checkpoints_dir:
        save_dir = os.path.join(
            args.checkpoints_dir,
            f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}.{x}",
        )
    else:
        save_dir = args.force_checkpoints_dir

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        if not dry_run(f"create directory: {save_dir}"):
            os.makedirs(save_dir)
            os.chmod(save_dir, 0o777)

    # if has_started(save_dir) and not args.resume_checkpoints_dir:
    #    print(f'skip in progress run: {save_dir}')
    #    return

    # generate train command
    cmd_args = []
    if args.data:
        cmd_args += [args.data]
    cmd_args += ["--save-dir", save_dir]
    for hp in config.values():
        cmd_args.extend(map(str, hp.get_cli_args()))
    cmd_args_str = " ".join(map(shlex.quote, cmd_args))
    if args.dry_run:
        dry_run(f"train command: fb_train.par {cmd_args_str}")

    # initialize train log
    train_log = os.path.join(save_dir, "train.log")
    if not dry_run(f"create train.log at: {train_log}"):
        with open(train_log, "a") as train_log_h:
            train_log_h.write("")
        os.chmod(train_log, 0o777)

    # write script
    script = get_script(
        port=get_random_port(),
        world_size=(args.num_nodes * args.num_gpus),
        train_fbpkg=train_fbpkg,
        cmd_args_str=cmd_args_str,
        stdout=train_log,
        stderr_prefix=os.path.join(save_dir, "train.stderr"),
        baseline_model_src=args.baseline_model,
        baseline_model_dst=os.path.join(save_dir, "checkpoint_last.pt"),
    )
    with tempfile.NamedTemporaryFile("w") as h:
        if not dry_run(f"write script to: {h.name}\n\n{script}"):
            h.write(script)
            h.flush()

        # crun
        crun_cmd = [
            "/usr/local/chronos/scripts/crun",
            "--print-url",
            "--mailwhen",
            "onFailure",
            "--hostgroup",
            str(args.hostgroup),
            "--gang-size",
            str(args.num_nodes),
            "-G",
            str(args.num_gpus),
            "-C",
            str(10 * args.num_gpus),
            "-M",
            ("-1" if args.num_gpus == 8 else str(29 * args.num_gpus)),
            #'--host-filter', 'gpu_model=="Tesla V100-SXM2-16GB"',
            h.name,
        ]
        crun_cmd_str = " ".join(map(shlex.quote, crun_cmd))

        env = os.environ.copy()
        if args.local:
            assert (
                args.num_nodes == 1
            ), "distributed training cannot be combined with --local"
            if not dry_run("start training locally"):
                if "CUDA_VISIBLE_DEVICES" not in env:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(
                        map(str, range(args.num_gpus))
                    )
                with tempfile.TemporaryDirectory() as tmpdir:
                    os.chmod(tmpdir, 0o777)
                    subprocess.Popen(
                        [
                            os.path.join(
                                "/data/users",
                                os.environ["USER"],
                                "fbsource/fbcode",
                                "buck-out/gen/deeplearning/projects/fairseq-py/fb_train.par",
                            )
                        ]
                        + cmd_args,
                        env=env,
                        cwd=tmpdir,
                    ).wait()
        else:
            if args.dry_run:
                print("| dry-run: start remote training")
                print(f"| dry-run: - run command: {crun_cmd_str}")
            else:
                subprocess.Popen(crun_cmd).wait()

    return train_log


def get_script(
    port,
    world_size,
    train_fbpkg,
    cmd_args_str,
    stdout,
    stderr_prefix,
    baseline_model_src,
    baseline_model_dst,
):
    if baseline_model_src is not None:
        link_baseline = f"""
        if [ ! -e {baseline_model_dst} ]; then
            cp {baseline_model_src} {baseline_model_dst}.tmp
            mv {baseline_model_dst}.tmp {baseline_model_dst}
        fi
        """
        wait_baseline = f"""
        while [ ! -e {baseline_model_dst} ]; do
            sleep 5
        done
        """
    else:
        link_baseline = ":"
        wait_baseline = ":"

    node_size = world_size if world_size < 8 else 8

    if world_size > 1:
        distributed = """\
        --distributed-init-method zeus://$CHRONOS_JOB_ID \
        --distributed-world-size $WORLD_SIZE \
        --distributed-rank $RANK
        """
    else:
        distributed = ""

    save_dir = os.path.dirname(baseline_model_dst)

    return f"""#!/bin/bash
/usr/local/bin/fbpkg fetch {train_fbpkg}

#if [ $(nvidia-smi | grep "No running processes found" | wc -l) != "1" ]; then
#    echo "Error: there are other running GPU processes"
#    exit 1
#fi

export MASTER_ADDR=$(/usr/local/chronos/scripts/clist -F name,hostname -n | grep $(echo $CHRONOS_JOB_NAME | sed "s/_GANG_MEMBER$//") | cut -d' ' -f 3).facebook.com
export MASTER_PORT={port}
export WORLD_SIZE={world_size}
export RANK=$(({node_size}*CHRONOS_GANG_MEMBER_ID))

echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo WORLD_SIZE: $WORLD_SIZE
echo RANK: $RANK

export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=8
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_BUFFSIZE=16777216 

# disable trees
export NCCL_TREE_THRESHOLD=0

## disable libgpumon
#export CUDA_INJECTION64_PATH=none

nvidia-smi

export

ulimit -a

ifconfig

ping6 -c 5 $MASTER_ADDR

if [ $RANK -eq 0 ]; then
    {link_baseline}
else
    {wait_baseline}
fi

mkdir -p {save_dir}
chmod 777 {save_dir}

LD_LIBRARY_PATH=/mnt/vol/gfsai-flash3-east/ai-group/users/myleott/nccl_2.4.8-1:$LD_LIBRARY_PATH ./fb_train.par {cmd_args_str} {distributed}
"""


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
