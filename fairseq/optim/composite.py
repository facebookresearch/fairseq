# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer, _build_optimizer
from fairseq.optim.lr_scheduler import FairseqLRScheduler, build_lr_scheduler
from omegaconf import II, open_dict
import copy


logger = logging.getLogger(__name__)


@dataclass
class OptimizerAndSchedulerConfig(FairseqDataclass):
    optimizer: Any = None
    lr_scheduler: Optional[Any] = None
    lr: List = II("optimization.lr")
    lr_float: Optional[
        float
    ] = None  # this makes it easier to sweep on learning rate with auto sweepers


@dataclass
class CompositeOptimizerConfig(FairseqDataclass):
    groups: Dict[str, Any] = field(
        default_factory=lambda: {},
        metadata={
            "help": "optimizer name -> optimizer OptimizerAndSchedulerConfig. "
            "Configures a different optimizer and (optionally) lr scheduler for each parameter group"
        },
    )
    dynamic_groups: bool = field(
        default=False,
        metadata={
            "help": "create groups dynamically based on parameters, if set to False, all parameters needs to have group_names"
        },
    )


@register_optimizer("composite", dataclass=CompositeOptimizerConfig)
class FairseqCompositeOptimizer(FairseqOptimizer):

    optimizers: Dict[str, FairseqOptimizer] = {}
    lr_schedulers: Dict[str, FairseqLRScheduler] = {}
    lr_scheduler: FairseqLRScheduler = None
    _optimizer: torch.optim.Optimizer

    def __init__(self, cfg: CompositeOptimizerConfig, params):
        super().__init__(cfg)

        assert (
            len(params) > 1
        ), "Composite optimizer only works when there are multiple parameter groups (try fp16_no_flatten_grads: true)"

        def dict_hash(dictionary: Dict[str, Any]) -> str:
            import hashlib
            import json

            dhash = hashlib.md5()
            encoded = json.dumps(dictionary, sort_keys=True).encode()
            dhash.update(encoded)
            return dhash.hexdigest()

        groupped_params = defaultdict(list)
        overrides = defaultdict(dict)
        if not cfg.dynamic_groups:
            for p in params:
                group = getattr(p, "param_group", "default")
                override_config = getattr(p, "optim_overrides", None)
                if override_config is not None and bool(override_config):
                    overrides[group] = override_config
                else:
                    assert (
                        override_config == None or override_config == overrides[group]
                    ), f"For group {group}, different overrides found {override_config} v/s {overrides[group]}"
                groupped_params[group].append(p)

            for p, params in groupped_params.items():
                override_config = getattr(params[0], "optim_overrides", None)
                if override_config is not None:
                    for pp in params[1:]:
                        assert override_config == getattr(
                            pp, "optim_overrides", None
                        ), f" {str(override_config)} != {str(getattr(pp, 'optim_overrides', None))}"
        else:
            for p in params:
                group = getattr(p, "param_group", "default")
                override_config = getattr(p, "optim_overrides", None)
                if override_config is not None:
                    override_config["group_name"] = group
                    group_name = dict_hash(override_config)
                    overrides[group_name] = override_config
                else:
                    group_name = group
                groupped_params[group_name].append(p)

        self.optimizers_config = {}
        for group, group_params in groupped_params.items():
            p_group = group
            if group in overrides and "group_name" in overrides[group]:
                p_group = overrides[group]["group_name"]
            if group in cfg.groups:
                group_cfg = cfg.groups[group]
                optimizer_config = copy.deepcopy(group_cfg.optimizer)
                scheduler_config = copy.deepcopy(group_cfg.lr_scheduler)
                explicit_group_present = True
            else:
                group_cfg = cfg.groups[p_group]
                optimizer_config = copy.deepcopy(group_cfg.optimizer)
                scheduler_config = copy.deepcopy(group_cfg.lr_scheduler)
                explicit_group_present = False

            if getattr(group_cfg, "lr_float", None) is not None:
                with open_dict(optimizer_config):
                    optimizer_config.lr = [group_cfg.lr_float]

            if group in overrides and "optimizer" in overrides[group]:
                with open_dict(optimizer_config):
                    if "lr_scale" in overrides[group]["optimizer"]:
                        lr_scale = overrides[group]["optimizer"]["lr_scale"]
                        optimizer_config.lr = [
                            lr * lr_scale for lr in optimizer_config.lr
                        ]

                        if explicit_group_present:
                            logger.info(
                                f"For group:{group}, config as well as override present for lr"
                            )

                    if (
                        "weight_decay_scale" in overrides[group]["optimizer"]
                        and "optimizer_config" in optimizer_config
                    ):
                        weight_decay_scale = overrides[group]["optimizer"][
                            "weight_decay_scale"
                        ]
                        optimizer_config.weight_decay = (
                            optimizer_config.weight_decay * weight_decay_scale
                        )
                        if explicit_group_present:
                            logger.info(
                                f"For group:{group}, config as well as override present for weight_decay"
                            )

            with open_dict(scheduler_config):
                scheduler_config.lr = optimizer_config.lr
            self.optimizers[group] = _build_optimizer(optimizer_config, group_params)
            self.optimizers_config[group] = optimizer_config
            if scheduler_config is not None:
                self.lr_schedulers[group] = build_lr_scheduler(
                    scheduler_config, self.optimizers[group]
                )
        logger.info("Optimizers for different groups are as below")
        for group in self.optimizers_config.keys():
            logger.info(f"Group : {group}:{self.optimizers_config[group]}")
        if len(self.lr_schedulers) > 0:
            assert len(self.lr_schedulers) == len(self.optimizers), (
                f"Please provide an lr scheduler for each optimizer to use pass_through scheduler. "
                f"Optimizers: {self.optimizers}; Lr scheds: {self.lr_schedulers}"
            )
            self.lr_scheduler = CompositeLRScheduler(self.lr_schedulers)

        self._optimizer = CompositeOptimizer(self.optimizers)

    @property
    def supports_groups(self):
        return True

    @property
    def param_groups(self):
        for opt in self.optimizers.values():
            for group in opt.param_groups:
                yield group

    def get_lr(self):
        """Return the current learning rate."""
        k = (
            "default"
            if "default" in self.optimizers
            else next(iter(self.optimizers.keys()))
        )
        return self.optimizers[k].param_groups[0]["lr"]

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {k: s.state_dict() for k, s in self.optimizers.items()}

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an LR scheduler state dict."""
        for k, state in state_dict.items():
            if k not in self.optimizers:
                # skip extra keys like "loss_scale" added by fp16 optimizer
                continue

            overrides = (
                optimizer_overrides[k]
                if isinstance(optimizer_overrides, dict) and k in optimizer_overrides
                else None
            )
            self.optimizers[k].load_state_dict(state, optimizer_overrides=overrides)


class CompositeOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers: Dict[str, FairseqOptimizer]):
        self.optimizers = optimizers

    @property
    def supports_memory_efficient_fp16(self):
        return all(o.supports_memory_efficient_fp16 for o in self.optimizers.values())

    @property
    def supports_flat_params(self):
        return all(o.supports_flat_params for o in self.optimizers.values())

    def step(self, closure=None, groups=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for k, opt in self.optimizers.items():
            if groups is None or k in groups:
                opt.step()

        return loss

    def zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad()


class CompositeLRScheduler(FairseqLRScheduler):
    def __init__(self, lr_schedulers):
        super().__init__(None, None)

        self.lr_schedulers = lr_schedulers

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {k: s.state_dict() for k, s in self.lr_schedulers.items()}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        for k, state in state_dict.items():
            self.lr_schedulers[k].load_state_dict(state)

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        for s in self.lr_schedulers.values():
            s.step_begin_epoch(epoch)

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        for s in self.lr_schedulers.values():
            s.step(epoch)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return {k: s.step_update(num_updates) for k, s in self.lr_schedulers.items()}
