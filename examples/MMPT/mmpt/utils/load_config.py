# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import omegaconf
from omegaconf import OmegaConf


def load_config(args=None, config_file=None, overwrite_fairseq=False):
    """TODO (huxu): move fairseq overwrite to another function."""
    if args is not None:
        config_file = args.taskconfig
    config = recursive_config(config_file)

    if config.dataset.subsampling is not None:
        batch_size = config.fairseq.dataset.batch_size // config.dataset.subsampling
        print(
            "adjusting batch_size to {} due to subsampling {}.".format(
                batch_size, config.dataset.subsampling
            )
        )
        config.fairseq.dataset.batch_size = batch_size

    is_test = config.dataset.split is not None and config.dataset.split == "test"
    if not is_test:
        if (
            config.fairseq.checkpoint is None
            or config.fairseq.checkpoint.save_dir is None
        ):
            raise ValueError("fairseq save_dir or save_path must be specified.")

        save_dir = config.fairseq.checkpoint.save_dir
        os.makedirs(save_dir, exist_ok=True)
        if config.fairseq.common.tensorboard_logdir is not None:
            tb_run_dir = suffix_rundir(
                save_dir, config.fairseq.common.tensorboard_logdir
            )
            config.fairseq.common.tensorboard_logdir = tb_run_dir
            print(
                "update tensorboard_logdir as", config.fairseq.common.tensorboard_logdir
            )
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(config=config, f=os.path.join(save_dir, "config.yaml"))

    if overwrite_fairseq and config.fairseq is not None and args is not None:
        # flatten fields.
        for group in config.fairseq:
            for field in config.fairseq[group]:
                print("overwrite args." + field, "as", config.fairseq[group][field])
                setattr(args, field, config.fairseq[group][field])
    return config


def recursive_config(config_path):
    """allows for stacking of configs in any depth."""
    config = OmegaConf.load(config_path)
    if config.includes is not None:
        includes = config.includes
        config.pop("includes")
        base_config = recursive_config(includes)
        config = OmegaConf.merge(base_config, config)
    return config


def suffix_rundir(save_dir, run_dir):
    max_id = -1
    for search_dir in os.listdir(save_dir):
        if search_dir.startswith(run_dir):
            splits = search_dir.split("_")
            cur_id = int(splits[1]) if len(splits) > 1 else 0
            max_id = max(max_id, cur_id)
    return os.path.join(save_dir, run_dir + "_" + str(max_id + 1))


def overwrite_dir(config, replace, basedir):
    for key in config:
        if isinstance(config[key], str) and config[key].startswith(basedir):
            config[key] = config[key].replace(basedir, replace)
        if isinstance(config[key], omegaconf.dictconfig.DictConfig):
            overwrite_dir(config[key], replace, basedir)
