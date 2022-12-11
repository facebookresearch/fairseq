# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf


@dataclass
class DependencySubmititConf(SlurmQueueConf):
    """Slurm configuration overrides and specific parameters"""

    _target_: str = (
        "hydra_plugins.dependency_submitit_launcher.launcher.DependencySubmititLauncher"
    )


ConfigStore.instance().store(
    group="hydra/launcher",
    name="dependency_submitit_slurm",
    node=DependencySubmititConf(),
    provider="dependency_submitit_slurm",
)
