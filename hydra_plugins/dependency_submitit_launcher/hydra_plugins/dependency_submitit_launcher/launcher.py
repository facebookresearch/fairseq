# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, List, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

from .config import DependencySubmititConf
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher


class DependencySubmititLauncher(BaseSubmititLauncher):
    _EXECUTOR = "slurm"

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:

        # lazy import to ensure plugin discovery remains fast
        import submitit

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0

        next_script = None

        for jo in job_overrides:
            if next_script is None:
                for item in jo:
                    if "next_script=" in item:
                        next_script = item
                        break
            assert (
                next_script is not None
            ), "job overrides must contain +next_script=path/to/next/script"
            jo.remove(next_script)

        idx = next_script.find("=")
        next_script = next_script[idx + 1 :]

        params = self.params
        # build executor
        init_params = {"folder": self.params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}

        init_params.update(
            **{
                f"{self._EXECUTOR}_{x}": y
                for x, y in params.items()
                if x in specific_init_keys
            }
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        # specify resources/parameters
        baseparams = set(OmegaConf.structured(DependencySubmititConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys
        }
        executor.update_parameters(**params)

        log.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : "
            f"{self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: List[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )

        jobs = executor.map_array(self, *zip(*job_params))

        for j, jp in zip(jobs, job_params):
            job_id = str(j.job_id)
            task_id = "0" if "_" not in job_id else job_id.split("_")[1]
            sweep_config = self.config_loader.load_sweep_config(self.config, jp[0])
            dir = sweep_config.hydra.sweep.dir

            dir = (
                dir.replace("[", "")
                .replace("]", "")
                .replace("{", "")
                .replace("}", "")
                .replace(",", "_")
                .replace("'", "")
                .replace('"', "")
            )

            subprocess.call(
                [next_script, job_id, task_id, dir],
                shell=False,
            )

        return [j.results()[0] for j in jobs]
