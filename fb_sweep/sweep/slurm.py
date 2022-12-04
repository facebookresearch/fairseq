import datetime
import hashlib
import itertools
import os
import random
import shlex
import shutil
import subprocess
import textwrap
from collections import OrderedDict
from glob import iglob

BASH_IF_CLAUSE = """
if [ "$SLURM_ARRAY_TASK_ID" = "{index}" ]; then
    {srun_cmd}
fi
"""


def main(get_grid, postprocess_hyperparams, args):
    def dry_run(msg):
        if args.dry_run:
            print(f"| dry-run:  {msg}")
        return args.dry_run

    if args.local:
        args.num_nodes = 1

    # compute all possible hyperparameter configurations
    grid = get_grid(args)
    grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    random.shuffle(grid_product)

    launch_train(args, grid, grid_product, dry_run, postprocess_hyperparams)


def copy_all_python_files(
    source, snapshot_main_dir, code_snapshot_hash, recurse_dirs="fairseq"
):
    """
    Copies following files from source to destination:
        a) all *.py files at direct source location.
        b) all fairseq/*.py recursively (default); recurse through comma-separated recurse_dirs
    """

    def all_pys(recurse_dirs):
        yield from iglob(os.path.join(source, "*.py"))
        for d in recurse_dirs.split(","):
            yield from iglob(os.path.join(source, d, "**/*.py"), recursive=True)
            yield from iglob(os.path.join(source, d, "**/*.so"), recursive=True)
            yield from iglob(os.path.join(source, d, "**/*.yaml"), recursive=True)

    os.makedirs(snapshot_main_dir, exist_ok=True)
    destination = os.path.join(snapshot_main_dir, code_snapshot_hash)
    assert not os.path.exists(destination), "Code snapshot: {0} alredy exists".format(
        code_snapshot_hash
    )
    os.makedirs(destination)

    for filepath in all_pys(recurse_dirs):
        directory, filename = os.path.split(filepath)
        if directory:
            os.makedirs(os.path.join(destination, directory), exist_ok=True)
        shutil.copy2(
            os.path.join(source, filepath), os.path.join(destination, filepath)
        )
    return destination


def run_setup(args, config, dry_run):
    # compute save_dir
    save_dir_key = ".".join(
        filter(
            lambda save_dir_key: save_dir_key is not None,
            [hp.get_save_dir_key() for hp in config.values()],
        )
    )
    save_dir_key = save_dir_key.replace(",", "_")
    num_total_gpus = args.num_nodes * args.num_gpus
    if args.use_jobarray:
        save_dir = os.path.join(
            args.checkpoints_dir,
            args.jobarray_name,
            f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}",
        )
    else:
        save_dir = os.path.join(
            args.checkpoints_dir, f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}"
        )

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        if not dry_run(f"create directory: {save_dir}"):
            os.makedirs(save_dir)

        # copy baseline model
        checkpoint_last = os.path.join(save_dir, "checkpoint_last.pt")
        if (
            args.baseline_model
            and not os.path.exists(checkpoint_last)
            and not dry_run(f"initialize with baseline model: {args.baseline_model}")
        ):
            if not os.path.exists(args.baseline_model):
                raise FileNotFoundError(
                    f"Cannot find baseline model: {args.baseline_model}"
                )
            shutil.copyfile(args.baseline_model, checkpoint_last)

    # create slurm log dir for job arrays
    if args.use_jobarray:
        slurm_dir = os.path.join(args.checkpoints_dir, args.jobarray_name, "slurm_logs")
        if not os.path.exists(slurm_dir):
            if not dry_run(f"create directory: {slurm_dir}"):
                os.makedirs(slurm_dir)
        return save_dir_key, save_dir, slurm_dir
    else:
        return save_dir_key, save_dir


def is_job_valid(args, save_dir, dry_run):
    # check for whether the run failed
    if has_finished(save_dir):
        if args.resume_finished:
            dry_run(f"restart previously finished run: {save_dir}")
        else:
            print(f"skip finished run (override with --resume-finished): {save_dir}")
            return False
    elif has_failed(save_dir):
        if args.resume_failed:
            dry_run(f"resume failed run: {save_dir}")
        else:
            print(f"skip failed run (override with --resume-failed): {save_dir}")
            return False
    elif has_started(save_dir):
        print(f"skip in progress run: {save_dir}")
        return False
    return True


def set_env(args, env, dry_run):
    if "OMP_NUM_THREADS" not in env:
        env["OMP_NUM_THREADS"] = "2"
    if args.local:
        if not dry_run("start training locally"):
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.num_gpus)))
            env["NCCL_DEBUG"] = "INFO"
    else:
        if args.num_nodes > 1:
            env["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
            env["NCCL_DEBUG"] = "INFO"


def gen_train_command(args, env, config, destination, save_dir, save_dir_key):
    # generate train command
    train_cmd = [args.python, os.path.join(destination, args.script)]
    train_cmd.extend(["--distributed-world-size", str(args.num_nodes * args.num_gpus)])
    if args.num_nodes > 1:
        train_cmd.extend(
            [
                "--distributed-port",
                str(get_random_port()),
            ]
        )
    if args.data is not None:
        train_cmd.extend([args.data])
    train_cmd.extend(["--save-dir", save_dir])
    if not args.no_tensorboard:
        _dir = args.tensorboard_logdir
        if _dir is None:
            _dir = os.path.join(
                "/checkpoint",
                env["USER"],
                "tensorboard_logs",
                str(datetime.date.today()),
            )
        tensorboard_logdir = os.path.join(
            _dir,
            f"{args.prefix}.{save_dir_key}.ngpu{str(args.num_nodes * args.num_gpus)}",
        )
        train_cmd.extend(["--tensorboard-logdir", tensorboard_logdir])
    if not args.no_wandb:
        if "WANDB_API_KEY" in env and "WANDB_BASE_URL" in env:
            if "--wandb-project" not in config:
                project = str(datetime.date.today())
                train_cmd.extend(["--wandb-project", project])
            if "WANDB_RUN_GROUP" not in env:
                env["WANDB_RUN_GROUP"] = args.prefix
            if "WANDB_RUN_ID" not in env:
                env["WANDB_RUN_ID"] = hashlib.md5(
                    os.path.basename(save_dir).encode("utf-8")
                ).hexdigest()
    for hp in config.values():
        train_cmd.extend(map(str, hp.get_cli_args()))
    return train_cmd


def gen_post_commands(args, save_dir):
    post_cmds = []
    if args.post_steps:
        for post_step in args.post_steps:
            if os.path.isfile(post_step):
                from pathlib import Path

                post_cmd = Path(post_step).read_text()
            else:
                post_cmd = post_step
            post_cmd = post_cmd.strip().format(
                job_dir=save_dir
            )  # assume to provide job_dir
            post_cmds.append(post_cmd)
    return post_cmds


def gen_srun_command_and_str(
    args, env, save_dir_key, train_log, train_stderr, train_cmd, post_cmds
):
    base_srun_cmd = [
        "srun",
        "--job-name",
        f"{args.prefix}.{save_dir_key}",
        "--output",
        train_log,
        "--error",
        train_stderr,
        "--open-mode",
        "append",
        "--unbuffered",
    ]
    if args.salloc:
        excluded_hosts = os.environ.get("EXCLUDED_HOSTS", None)
        included_hosts = os.environ.get("INCLUDED_HOSTS", None)
        base_srun_cmd += [
            "--nodes",
            str(args.num_nodes),
            "--ntasks",
            str(args.num_nodes),
        ]
        base_srun_cmd += ["-x", excluded_hosts] if excluded_hosts is not None else []
        base_srun_cmd += ["-w", included_hosts] if included_hosts is not None else []

    srun_cmd = base_srun_cmd + train_cmd
    srun_cmd_str = " ".join(map(shlex.quote, srun_cmd))
    for post_cmd in post_cmds:
        post_cmd_str = " ".join(map(shlex.quote, base_srun_cmd)) + f" {post_cmd}"
        srun_cmd_str = (
            f"({srun_cmd_str} && {post_cmd_str})"
            if len(srun_cmd_str) > 0
            else post_cmd_str
        )
    return srun_cmd, srun_cmd_str


def gen_sbatch_command_and_str(
    args,
    job_name,
    train_log,
    train_stderr,
    destination,
    srun_cmd_str,
    array_length=None,
):
    excluded_hosts = os.environ.get("EXCLUDED_HOSTS", None)
    included_hosts = os.environ.get("INCLUDED_HOSTS", None)
    sbatch_cmd = [
        "sbatch",
        "--job-name",
        job_name,
        "--gpus",
        str(args.num_gpus * args.num_nodes),
        "--nodes",
        str(args.num_nodes),
        "--ntasks-per-node",
        "1",
        "--cpus-per-task",
        str(int(8 * args.num_gpus)),
        "--output",
        train_log,
        "--error",
        train_stderr,
        "--open-mode",
        "append",
        # '--no-requeue',
        "--signal",
        "B:USR1@180",
    ]
    if array_length is not None:
        sbatch_cmd += ["--array", f"0-{array_length-1}"]

    if args.constraint:
        sbatch_cmd += ["--constraint", args.constraint]

    if args.partition:
        sbatch_cmd += ["--partition", args.partition]
    if args.reservation:
        sbatch_cmd += ["--reservation", args.reservation]
    if args.exclusive:
        sbatch_cmd += ["--exclusive"]
    if args.comment:
        comment = args.comment
        if args.snapshot_code:
            comment += ", Code Location: {0}".format(destination)
        sbatch_cmd += ["--comment", comment]
    elif args.snapshot_code:
        sbatch_cmd += ["--comment", "Code Location: {0}".format(destination)]

    if args.dep is not None:
        sbatch_cmd.extend(["-d", str(args.dep)])
    if args.time is not None:
        sbatch_cmd.extend(["--time", args.time])
    if args.mem is not None:
        sbatch_cmd += ["--mem", args.mem]
    else:
        sbatch_cmd += ["--mem-per-cpu", "7G"]
    sbatch_cmd += ["-x", excluded_hosts] if excluded_hosts is not None else []
    sbatch_cmd += ["-w", included_hosts] if included_hosts is not None else []

    wrapped_cmd = requeue_support() + "\n" + srun_cmd_str
    if array_length is None:
        wrapped_cmd = wrapped_cmd + " \n wait $! \n sleep 610 & \n wait $!"

    sbatch_cmd += ["--wrap", wrapped_cmd]
    sbatch_cmd_str = " ".join(map(shlex.quote, sbatch_cmd))
    return sbatch_cmd, sbatch_cmd_str


def local_run(args, env, train_cmd, post_cmds, dry_run):
    assert args.num_nodes == 1, "distributed training cannot be combined with --local"
    if not dry_run("start training locally"):
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.num_gpus)))
        env["NCCL_DEBUG"] = "INFO"
        train_proc = subprocess.Popen(train_cmd, env=env)
        train_proc.wait()
        for post_cmd in post_cmds:
            post_cmd_proc = subprocess.Popen(post_cmd, shell=True, env=env)
            post_cmd_proc.wait()


def run_batch(env, sbatch_cmd_str, sbatch_cmd):
    print(f"running command: {sbatch_cmd_str}\n")
    with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
        stdout = train_proc.stdout.read().decode("utf-8")
        try:
            job_id = int(stdout.rstrip().split()[-1])
            print(f"Launched job {job_id}")
        except IndexError:
            job_id = None
    return job_id, stdout


def write_git_commit(args, train_log):
    with open(train_log, "a") as train_log_h:
        # log most recent git commit
        git_commit = subprocess.check_output(
            "git log | head -n 1", shell=True, encoding="utf-8"
        )
        print(git_commit.rstrip(), file=train_log_h)
        if args.baseline_model:
            print(f"baseline model: {args.baseline_model}", file=train_log_h)


def dry_run_batch(env, train_log, train_stderr, sbatch_cmd_str, sbatch_cmd, dry_run):
    dry_run("start remote training")
    dry_run(f"- log stdout to: {train_log}")
    dry_run(f"- log stderr to: {train_stderr}")
    dry_run(f"- run command: {sbatch_cmd_str}")
    sbatch_cmd += ["--test-only"]
    with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
        stdout = train_proc.stdout.read().decode("utf-8")
        print(stdout)


def launch_train(args, grid, grid_product, dry_run, postprocess_hyperparams):
    destination = ""
    if args.snapshot_code:
        # Currently hash is just the current time in ISO format.
        # Remove colons since they cannot be escaped in POSIX PATH env vars.
        code_snapshot_hash = datetime.datetime.now().isoformat().replace(":", "_")
        destination = copy_all_python_files(
            ".",
            os.path.join(args.snapshot_root, "slurm_snapshot_code"),
            code_snapshot_hash,
            args.snapshot_recurse_dirs,
        )
        os.environ["PYTHONPATH"] = destination + ":" + os.environ.get("PYTHONPATH", "")

    # set environment
    env = os.environ.copy()
    set_env(args, env, dry_run)

    # start training
    srun_cmd_str_list = []
    train_log_list = []
    for i, hp_values in enumerate(grid_product):
        if i == args.num_trials:
            break
        config = OrderedDict()
        for hp, value in zip(grid, hp_values):
            config[hp.name] = hp
            config[hp.name].current_value = value

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        if args.use_jobarray:
            save_dir_key, save_dir, slurm_dir = run_setup(args, config, dry_run)
        else:
            save_dir_key, save_dir = run_setup(args, config, dry_run)

        # check if job failed, exists, finished
        if not is_job_valid(args, save_dir, dry_run):
            continue

        # generate train command
        train_cmd = gen_train_command(
            args, env, config, destination, save_dir, save_dir_key
        )

        # post cmds
        post_cmds = gen_post_commands(args, save_dir)

        train_log = os.path.join(save_dir, "train.log")
        train_stderr = os.path.join(save_dir, "train.stderr.%j")  # %j = slurm job id
        srun_cmd, srun_cmd_str = gen_srun_command_and_str(
            args, env, save_dir_key, train_log, train_stderr, train_cmd, post_cmds
        )

        # launch each job individually
        if not args.use_jobarray:
            job_id = None
            if args.dry_run:
                train_cmd_str = " ".join(train_cmd)
                dry_run(f"train command: {train_cmd_str}")
                for post_cmd in post_cmds:
                    dry_run(f"post steps command: {post_cmd}")
            if args.local:
                local_run(args, env, train_cmd, post_cmds, dry_run)
            else:
                srun_cmd_str = srun_cmd_str + " &"
                # build command
                if not args.salloc:
                    job_name = f"{args.prefix}.{save_dir_key}"
                    sbatch_cmd, sbatch_cmd_str = gen_sbatch_command_and_str(
                        args,
                        job_name,
                        train_log,
                        train_stderr,
                        destination,
                        srun_cmd_str,
                    )
                else:
                    sbatch_cmd = srun_cmd
                    sbatch_cmd_str = srun_cmd_str
                if args.dry_run:
                    dry_run_batch(
                        env,
                        train_log,
                        train_stderr,
                        sbatch_cmd_str,
                        sbatch_cmd,
                        dry_run,
                    )
                else:
                    write_git_commit(args, train_log)
                    with open(train_log, "a") as train_log_h:
                        job_id, stdout = run_batch(env, sbatch_cmd_str, sbatch_cmd)
                        print(stdout, file=train_log_h)
            if job_id is not None:
                print("Launched {}".format(job_id))
            if args.sequential and not args.local and job_id is not None:
                args.dep = job_id
        else:
            train_log_list.append(train_log)
            srun_cmd_str_list.append(srun_cmd_str)

            if not args.dry_run:
                write_git_commit(args, train_log)
    # aggregate cmds and launch single job array
    if args.use_jobarray:
        aggregate_cmd = ""
        for i, srun_cmd_str in enumerate(srun_cmd_str_list):
            aggregate_cmd = aggregate_cmd + BASH_IF_CLAUSE.format(
                index=i, srun_cmd=srun_cmd_str
            )
        job_name = args.jobarray_name
        slurm_stdout_log = os.path.join(slurm_dir, "slrm_stdout.%j")
        slurm_stderr_log = os.path.join(slurm_dir, "slrm_stderr.%j")
        array_length = len(srun_cmd_str_list)
        sbatch_cmd, sbatch_cmd_str = gen_sbatch_command_and_str(
            args,
            job_name,
            slurm_stdout_log,
            slurm_stderr_log,
            destination,
            aggregate_cmd,
            array_length=array_length,
        )

        if args.dry_run:
            dry_run_batch(
                env,
                slurm_stdout_log,
                slurm_stderr_log,
                sbatch_cmd_str,
                sbatch_cmd,
                dry_run,
            )
        else:
            job_id, stdout = run_batch(env, sbatch_cmd_str, sbatch_cmd)
            for train_log in train_log_list:
                with open(train_log, "a") as train_log_h:
                    print(stdout, file=train_log_h)


def has_finished(save_dir):
    train_log = os.path.join(save_dir, "train.log")
    if not os.path.exists(train_log):
        return False
    with open(train_log, "r") as h:
        lines = h.readlines()
        if len(lines) == 0:
            return False
        if "done training" in lines[-1]:
            return True
    return False


def has_failed(save_dir):
    if not os.path.exists(save_dir):
        return False

    # find max job id
    job_ids = []
    for fn in os.listdir(save_dir):
        if fn.startswith("train.stderr."):
            job_ids.append(int(fn.split(".")[-1]))
    if len(job_ids) == 0:
        return False
    max_job_id = max(job_ids)

    def _has_failed(stderr_fn):
        with open(stderr_fn, "r") as h:
            for line in h:
                if len(line.strip()) > 0:
                    # assume that any output in stderr indicates an error
                    return True
        return False

    return _has_failed(os.path.join(save_dir, f"train.stderr.{max_job_id}"))


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


def requeue_support():
    return textwrap.dedent(
        """
        trap_handler () {
           echo "Caught signal: " $1
           # SIGTERM must be bypassed
           if [ "$1" = "TERM" ]; then
               echo "bypass sigterm"
           else
             # Submit a new job to the queue
             echo "Requeuing " $SLURM_JOB_ID
             scontrol requeue $SLURM_JOB_ID
           fi
        }


        # Install signal handler
        trap 'trap_handler USR1' USR1
        trap 'trap_handler TERM' TERM
    """
    )
