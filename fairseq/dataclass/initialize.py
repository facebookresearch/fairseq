# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import logging
from typing import Dict, Any
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import FairseqConfig


logger = logging.getLogger(__name__)


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
