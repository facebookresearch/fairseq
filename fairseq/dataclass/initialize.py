# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import logging
from typing import Dict, Any
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import FairseqConfig

# the imports below are necessary so that "REGISTRIES" is correctly populated with all components
from fairseq.criterions import CRITERION_REGISTRY  # noqa
from fairseq.optim import OPTIMIZER_REGISTRY  # noqa
from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY  # noqa
from fairseq.scoring import SCORER_REGISTRY  # noqa
from fairseq.data.encoders import BPE_REGISTRY, TOKENIZER_REGISTRY  # noqa

from fairseq.models import MODEL_DATACLASS_REGISTRY
from fairseq.tasks import TASK_DATACLASS_REGISTRY
from fairseq.registry import REGISTRIES


logger = logging.getLogger(__name__)


def register_module_dataclass(
    cs: ConfigStore, registry: Dict[str, Any], group: str
) -> None:
    """register dataclasses defined in modules in config store, for example, in migrated tasks, models, etc."""
    # note that if `group == model`, we register all model archs, not the model name.
    for k, v in registry.items():
        node_ = v()
        node_._name = k
        cs.store(name=k, group=group, node=node_, provider="fairseq")


def hydra_init(cfg_name="config") -> None:

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=FairseqConfig)

    for k in FairseqConfig.__dataclass_fields__:
        v = FairseqConfig.__dataclass_fields__[k].default
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise

    register_module_dataclass(cs, TASK_DATACLASS_REGISTRY, "task")
    register_module_dataclass(cs, MODEL_DATACLASS_REGISTRY, "model")

    for k, v in REGISTRIES.items():
        register_module_dataclass(cs, v["dataclass_registry"], k)
