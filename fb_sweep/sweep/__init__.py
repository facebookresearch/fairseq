import argparse
import datetime
import os
import socket
from typing import List, Optional


def csv_str_list(x):
    return [y.strip() for y in x.split(",")]


def get_args(add_extra_options_func=None, input_args: Optional[List[str]] = None):
    """
    input_args (List[str]): strings to parse, defaults to sys.argv
    """
    parser = argparse.ArgumentParser("Script for launching hyperparameter sweeps ")
    parser.add_argument("--grid", help="grid function we used", default=None)
    parser.add_argument("--pair", help="language direction", default=None)

    parser.add_argument("-d", "--data", help="path to data directory")
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "-t",
        "--num-trials",
        required=True,
        type=int,
        help="number of random hyperparam configurations to try (-1 for grid search)",
    )
    parser.add_argument(
        "-g", "--num-gpus", type=int, required=True, help="number of GPUs per node"
    )
    parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--baseline-model", help="path to baseline model from which to resume training"
    )
    parser.add_argument(
        "--force-checkpoints-dir", help="force using a given checkpoint dir"
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="resume any runs that failed (assumes --num-trials and --seed are the same)",
    )
    parser.add_argument(
        "--resume-finished",
        action="store_true",
        help="force any runs that finished to begin again (uncommon)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="output only a list of actions to perform without performing them",
    )
    parser.add_argument("--local", action="store_true", help="run job locally")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--script", default="train.py", help="script to launch")
    parser.add_argument(
        "--python", default="python", help="path to nonstandard python binary"
    )

    try:
        import torch_xla  # noqa
        tpu = True
    except ImportError:
        tpu = False

    hostname = socket.gethostname()
    if "fair" in hostname:
        default_backend = "slurm"
        parser.add_argument(
            "--checkpoints-dir",
            default=os.path.join(
                "/checkpoint", os.environ["USER"], str(datetime.date.today())
            ),
            help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
        )
    elif tpu:
        default_backend = "tpu"
        parser.add_argument(
            "--checkpoints-dir",
            default=os.path.join(
                "/mnt/fairseq_data",
                os.environ["USER"],
                "checkpoints",
                str(datetime.date.today()),
            ),
            help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
        )
    else:
        default_backend = "fblearner"
        parser.add_argument(
            "--checkpoints-dir",
            default=os.path.join(
                "/mnt/vol/gfsai-east/ai-group/users",
                os.environ["USER"],
                "checkpoints",
                str(datetime.date.today()),
            ),
            help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
        )
        parser.add_argument(
            "--log-main-dir",
            default=None,
            help="dir to store log in addition to stdout. If this "
            "is not set, it will be set to args.checkpoints_dir",
        )

    parser.add_argument(
        "--backend",
        choices=["fblearner", "chronos", "slurm", "tpu"],
        default=default_backend,
    )

    # FBLearner params
    parser.add_argument("--entitlement", help="entitlement to use", default="gpu_fair")
    parser.add_argument(
        "--run-as-secure-group", help="secure group to use", default="oncall_fairseq"
    )
    parser.add_argument(
        "--capabilities",
        help="capabilities: e.g. GPU_V100_HOST,GPU_V100_32G_HOST",
        type=csv_str_list,
        default=None,
    )
    # for manifold in fblearner
    parser.add_argument(
        "--manifold-max-parallel",
        default=8,
        type=int,
        help="set ManifoldPathHandler max_parallel download number",
    )
    parser.add_argument(
        "--manifold-timeout-sec",
        default=1800,
        type=int,
        help="set ManifoldPathHandler timeout seconds",
    )
    parser.add_argument(
        "--manifold-has-user-data",
        default=None,
        type=lambda x: x.lower() not in ("no", "false", "f", "n", "0")
        if x is not None
        else None,
        help="set ManifoldPathHandler has_user_data option",
    )
    parser.add_argument(
        "--manifold-num-retries",
        default=15,
        type=int,
        help="set ManifoldPathHandler num_retries option",
    )
    parser.add_argument(
        "--manifold-ttl",
        default=None,
        type=int,
        help="A manifold  resource's time-to-live, applied to all manifold written resources. By default, there is no TTL.",
    )
    # for manifold in fblearner

    # Chronos params
    parser.add_argument("--hostgroup", help="hostgroup to use")
    parser.add_argument("--host-filter", help="host filter")
    parser.add_argument("--fbpkg", help="use the given fbpkg")
    parser.add_argument("--build-only", action="store_true")

    # Slurm params
    parser.add_argument(
        "--salloc", action="store_true", help="run agaist current allocation"
    )
    parser.add_argument("--partition", help="partition to run on", default="learnfair")
    parser.add_argument("--reservation", help="reservation to run on")
    parser.add_argument(
        "--exclusive", action="store_true", help="if set, get exclusive host"
    )
    parser.add_argument(
        "--dep",
        metavar="JOBID",
        type=int,
        help="add JOBID as a dependency (i.e., wait for it to finish)",
    )
    parser.add_argument(
        "--sequential", action="store_true", help="schedule jobs to run sequentially"
    )
    parser.add_argument(
        "--time", default="4320", help="expected job duration in minutes"
    )
    parser.add_argument("--mem", "--mem", help="memory to request")
    parser.add_argument(
        "--constraint",
        metavar="CONSTRAINT",
        help='gpu constraint, if any. e.g. "volta"',
    )
    parser.add_argument("--comment", help="comment string")
    parser.add_argument(
        "--snapshot-code",
        action="store_true",
        default=False,
        help="Flag for creating a snapshot of training code while creating slurm job,"
        ' path is "./slurm_snapshot_code/<TIME_ISO_FORMAT/>:", '
        "can find time from comment of slurm job.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default=".",
        help="root path for saving the snapshot code.",
    )
    parser.add_argument(
        "--snapshot-recurse-dirs",
        default="fairseq,fairseq_cli",
        help="comma-separated directories from where to recursively copy *.py, *.so and *.yaml files",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        help="save tensorboard logs in <tensorboard-logdir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="disable tensorboard logging"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="disable WandB logging"
    )
    parser.add_argument(
        "--post-steps",
        nargs="+",
        help="additional steps to execute after the primary job is complete. "
        "this can be a file with the steps, or a string. some placeholders such as "
        "{job_dir} will be replaced",
    )
    parser.add_argument('--use-jobarray', action='store_true', help="Submit sweep as job-array")
    parser.add_argument('--jobarray-name', type=str, default=None, help="Folder name for job-array. Defaults to <jobarray_timestamp>")

    # GCP params
    parser.add_argument("--tpu", help="tpu to use")

    if add_extra_options_func is not None:
        add_extra_options_func(parser)
    args = parser.parse_args(input_args)
    if args.use_jobarray:
        if args.jobarray_name is None:
            ja_hash = datetime.datetime.now().isoformat().replace(':', '_')
            args.jobarray_name = f'jobarray_{ja_hash}'
        assert not args.local, 'Job array should not be local'
        assert not args.sequential, 'Cannot have both sequential and jobarray'
    return args


class hyperparam(object):
    """Base class for defining hyperparameters."""

    def __init__(
        self,
        name,
        values=None,
        binary_flag=False,
        save_dir_key=None,
        positional_arg=False,
    ):
        """
        Arguments:
        - name : the name of the hyperparameter (e.g., `--dropout`)
        - values : the set of values to sweep over (e.g., `[0.0, 0.1, 0.2]`)
        - binary_flag : whether the hyperparameter uses a boolean flag (e.g., `--no-save`)
        - save_dir_key : function that takes the hyperparameter value and returns the "key"
                         to be appended to the output directory name
        - positional_arg : whether the hyperparameter is a positional argument
        """
        self.name = name
        if values is None:  # syntactic sugar for binary flags
            self.values = [True]
            self.binary_flag = True
        else:
            self.values = values if isinstance(values, list) else [values]
            self.binary_flag = binary_flag
        self.save_dir_key = save_dir_key
        self.positional_arg = positional_arg
        self.current_value = None

        if positional_arg and name.startswith("-"):
            raise ValueError(
                f"positional arguments must not start with a dash ({name})"
            )

        if len(self.values) > 1 and self.save_dir_key is None:
            raise ValueError(
                f"{name} has more than one value but is missing a save_dir_key!"
            )

    def get_cli_args(self):
        if self.binary_flag:
            return [self.name] if self.current_value else []
        elif self.positional_arg:
            return [self.current_value]
        else:
            return [self.name, self.current_value]

    def get_save_dir_key(self):
        if self.save_dir_key is None:
            return None
        if self.binary_flag:
            return self.save_dir_key(1) if self.current_value else None
        return self.save_dir_key(self.current_value)


def main(
    get_grid,
    postprocess_hyperparams,
    add_extra_options_func=None,
    scheduler_args: Optional[List[str]] = None,
):
    args = get_args(add_extra_options_func, scheduler_args)
    if args.backend == "fblearner":
        from .fblearner import main as backend_main
    elif args.backend == "chronos":
        from .chronos import main as backend_main
    elif args.backend == "slurm":
        from .slurm import main as backend_main
    elif args.backend == "tpu":
        from .tpu import main as backend_main

    get_grid = get_grid[args.grid] if args.grid is not None else get_grid
    backend_main(get_grid, postprocess_hyperparams, args)
