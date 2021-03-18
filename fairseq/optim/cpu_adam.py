# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, DictConfig
import logging


try:
    import deepspeed.op_extensions.cpu_adam as ds_opt_adam
    has_deepspeed_cpu_adam = True
except ImportError as e:
    logging.warning(e)
    has_deepspeed_cpu_adam = False


@dataclass
class FairseqCPUAdamConfig(FairseqDataclass):
    adam_betas: str = field(
        default="(0.9, 0.999)", metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    fp16_adam_stats: bool = field(
        default=False, metadata={"help": "use FP16 stats (with automatic scaling)"}
    )
    # TODO common vars below in parent
    lr: List[float] = II("optimization.lr")


@register_optimizer("cpu_adam", dataclass=FairseqCPUAdamConfig)
class FairseqCPUAdam(FairseqOptimizer):
    """Adam optimizer for fairseq, optimized for CPU tensors.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: DictConfig, params):
        super().__init__(cfg)
        self._optimizer = CPUAdam(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
            "use_fp16_stats": self.cfg.fp16_adam_stats,
        }


class CPUAdam(torch.optim.Optimizer):

    optimizer_id = 0

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        use_fp16_stats=False,
    ):
        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        self.use_fp16_stats = use_fp16_stats
        self.FLOAT16_MAX = 65504.0

        if not has_deepspeed_cpu_adam:
            raise ImportError("Please install DeepSpeed: pip install deepspeed")

        self.opt_id = CPUAdam.optimizer_id
        CPUAdam.optimizer_id = CPUAdam.optimizer_id + 1

        self.ds_opt_adam = ds_opt_adam
        adamw_mode = True
        self.ds_opt_adam.create_adam(
            self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode
        )

    @property
    def supports_flat_params(self):
        return True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    dtype = torch.float16 if self.use_fp16_stats else p.data.dtype
                    # gradient momentums
                    state["exp_avg"] = torch.zeros_like(
                        p.data, dtype=dtype, device="cpu"
                    )
                    # gradient variances
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, dtype=dtype, device="cpu"
                    )
                    if self.use_fp16_stats:
                        assert torch.is_floating_point(p.data)
                        state["exp_avg_scale"] = 1.0
                        state["exp_avg_sq_scale"] = 1.0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                p_data_bak = p.data  # backup of the original data pointer

                p.data = p.data.to(dtype=torch.float32, device="cpu")
                p.grad.data = p.grad.data.to(dtype=torch.float32, device="cpu")

                if self.use_fp16_stats:
                    exp_avg = exp_avg.float() * state["exp_avg_scale"]
                    exp_avg_sq = exp_avg_sq.float() * state["exp_avg_sq_scale"]

                state["step"] += 1
                beta1, beta2 = group["betas"]

                self.ds_opt_adam.adam_update(
                    self.opt_id,
                    state["step"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["weight_decay"],
                    group["bias_correction"],
                    p.data,
                    p.grad.data,
                    exp_avg,
                    exp_avg_sq,
                )

                if p_data_bak.data_ptr() != p.data.data_ptr():
                    p_data_bak.copy_(p.data)
                    p.data = p_data_bak

                if self.use_fp16_stats:

                    def inf_norm(t):
                        return torch.norm(t, float("inf"))

                    # from github.com/openai/jukebox/blob/master/jukebox/utils/fp16.py
                    state["exp_avg_scale"], state["exp_avg_sq_scale"] = (
                        1e-8 + inf_norm(exp_avg) / self.FLOAT16_MAX,
                        1e-8 + inf_norm(exp_avg_sq) / self.FLOAT16_MAX,
                    )
                    state["exp_avg"], state["exp_avg_sq"] = (
                        (exp_avg / state["exp_avg_scale"]).half(),
                        (exp_avg_sq / state["exp_avg_sq_scale"]).half(),
                    )

        return loss
