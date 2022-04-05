# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import logging
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import FairseqConfig
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def hydra_init(cfg_name="config") -> None:

    cs = ConfigStore.instance()
    cs.store(name=f"{cfg_name}", node=FairseqConfig)

    for k in FairseqConfig.__dataclass_fields__:
        v = FairseqConfig.__dataclass_fields__[k].default
        try:
            cs.store(name=k, node=v)
        except BaseException:
            logger.error(f"{k} - {v}")
            raise


def add_defaults(cfg: DictConfig) -> None:
    """This function adds default values that are stored in dataclasses that hydra doesn't know about"""

    from fairseq.registry import REGISTRIES
    from fairseq.tasks import TASK_DATACLASS_REGISTRY
    from fairseq.models import ARCH_MODEL_NAME_REGISTRY, MODEL_DATACLASS_REGISTRY
    from fairseq.dataclass.utils import merge_with_parent
    from typing import Any

    OmegaConf.set_struct(cfg, False)

    for k, v in FairseqConfig.__dataclass_fields__.items():
        field_cfg = cfg.get(k)
        if field_cfg is not None and v.type == Any:
            dc = None

            if isinstance(field_cfg, str):
                field_cfg = DictConfig({"_name": field_cfg})
                field_cfg.__dict__["_parent"] = field_cfg.__dict__["_parent"]

            name = getattr(field_cfg, "_name", None)

            if k == "task":
                dc = TASK_DATACLASS_REGISTRY.get(name)
            elif k == "model":
                name = ARCH_MODEL_NAME_REGISTRY.get(name, name)
                dc = MODEL_DATACLASS_REGISTRY.get(name)
            elif k in REGISTRIES:
                dc = REGISTRIES[k]["dataclass_registry"].get(name)

            if dc is not None:
                cfg[k] = merge_with_parent(dc, field_cfg)
