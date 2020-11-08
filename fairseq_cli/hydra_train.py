#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from omegaconf import OmegaConf
import os

from fairseq.dataclass.initialize import hydra_init
from fairseq_cli.train import main as pre_main
from fairseq import distributed_utils
from fairseq.dataclass.configs import FairseqConfig

import logging
import torch


logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> None:

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))

    OmegaConf.set_struct(cfg, True)

    if cfg.common.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, pre_main)
    else:
        distributed_utils.call_main(cfg, pre_main)


if __name__ == "__main__":
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    hydra_init(cfg_name)
    hydra_main()
