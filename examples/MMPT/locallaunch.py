# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os

from omegaconf import OmegaConf

from mmpt.utils import recursive_config, overwrite_dir
from mmpt_cli.localjob import LocalJob


class JobLauncher(object):
    JOB_CONFIG = {
        "local": LocalJob,
    }

    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        job_key = "local"

        if yaml_file.endswith(".yaml"):
            config = recursive_config(yaml_file)
            if config.task_type is not None:
                job_key = config.task_type.split("_")[0]
        else:
            raise ValueError("unknown extension of job file:", yaml_file)
        self.job_key = job_key

    def __call__(self, job_type=None, dryrun=False):
        if job_type is not None:
            self.job_key = job_type.split("_")[0]
        print("[JobLauncher] job_key", self.job_key)
        job = JobLauncher.JOB_CONFIG[self.job_key](
            self.yaml_file, job_type=job_type, dryrun=dryrun)
        return job.submit()


class Pipeline(object):
    """a job that loads yaml config."""

    def __init__(self, fn):
        """
        load a yaml config of a job and save generated configs as yaml for each task.
        return: a list of files to run as specified by `run_task`.
        """
        if fn.endswith(".py"):
            # a python command.
            self.backend = "python"
            self.run_yamls = [fn]
            return

        job_config = recursive_config(fn)
        if job_config.base_dir is None:  # single file job config.
            self.run_yamls = [fn]
            return

        self.project_dir = os.path.join("projects", job_config.project_dir)
        self.run_dir = os.path.join("runs", job_config.project_dir)

        if job_config.run_task is not None:
            run_yamls = []
            for stage in job_config.run_task:
                # each stage can have multiple tasks running in parallel.
                if OmegaConf.is_list(stage):
                    stage_yamls = []
                    for task_file in stage:
                        stage_yamls.append(
                            os.path.join(self.project_dir, task_file))
                    run_yamls.append(stage_yamls)
                else:
                    run_yamls.append(os.path.join(self.project_dir, stage))
            self.run_yamls = run_yamls
        configs_to_save = self._overwrite_task(job_config)
        self._save_configs(configs_to_save)

    def __getitem__(self, idx):
        yaml_files = self.run_yamls[idx]
        if isinstance(yaml_files, list):
            return [JobLauncher(yaml_file) for yaml_file in yaml_files]
        return [JobLauncher(yaml_files)]

    def __len__(self):
        return len(self.run_yamls)

    def _save_configs(self, configs_to_save: dict):
        # save
        os.makedirs(self.project_dir, exist_ok=True)
        for config_file in configs_to_save:
            config = configs_to_save[config_file]
            print("saving", config_file)
            OmegaConf.save(config=config, f=config_file)

    def _overwrite_task(self, job_config):
        configs_to_save = {}
        self.base_project_dir = os.path.join("projects", job_config.base_dir)
        self.base_run_dir = os.path.join("runs", job_config.base_dir)

        for config_sets in job_config.task_group:
            overwrite_config = job_config.task_group[config_sets]
            if (
                overwrite_config.task_list is None
                or len(overwrite_config.task_list) == 0
            ):
                print(
                    "[warning]",
                    job_config.task_group,
                    "has no task_list specified.")
            # we don't want this added to a final config.
            task_list = overwrite_config.pop("task_list", None)
            for config_file in task_list:
                config_file_path = os.path.join(
                    self.base_project_dir, config_file)
                config = recursive_config(config_file_path)
                # overwrite it.
                if overwrite_config:
                    config = OmegaConf.merge(config, overwrite_config)
                overwrite_dir(config, self.run_dir, basedir=self.base_run_dir)
                save_file_path = os.path.join(self.project_dir, config_file)
                configs_to_save[save_file_path] = config
        return configs_to_save


def main(args):
    job_type = args.jobtype if args.jobtype else None
    # parse multiple pipelines.
    pipelines = [Pipeline(fn) for fn in args.yamls.split(",")]

    for pipe_id, pipeline in enumerate(pipelines):
        if not hasattr(pipeline, "project_dir"):
            for job in pipeline[0]:
                job(job_type=job_type, dryrun=args.dryrun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yamls", type=str)
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="run config and prepare to submit without launch the job.",
    )
    parser.add_argument(
        "--jobtype", type=str, default="",
        help="force to run jobs as specified.")
    args = parser.parse_args()
    main(args)
