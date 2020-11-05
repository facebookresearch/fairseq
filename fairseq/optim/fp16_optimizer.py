# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from itertools import chain

import torch
from fairseq import optim
from omegaconf import DictConfig

from .dynamic_loss_scaler import DynamicLossScaler


class _FP16OptimizerMixin(object):
    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)
        self._multiply_factor = 1.0

    @property
    def has_flat_params(self):
        return torch.is_tensor(self.fp32_params) or (
            isinstance(self.fp32_params, dict)
            and all(torch.is_tensor(t) for t in self.fp32_params.values())
        )

    @classmethod
    def build_fp32_params(cls, args, params, flatten=True):
        # create FP32 copy of parameters and grads
        if flatten:
            is_pipeline_parallel = getattr(
                args, "pipeline_model_parallel", False
            ) and getattr(args, "distributed_no_spawn", False)
            total_param_size = sum(p.data.numel() for p in params)
            devices = [torch.cuda.current_device()]
            if is_pipeline_parallel:
                devices = list(set(args.pipeline_devices))
            fp32_params = {}
            for device in devices:
                if is_pipeline_parallel:
                    device_param_size = sum(
                        p.data.numel() for p in params if p.device.index == device
                    )
                    device_params = [p for p in params if p.device.index == device]
                else:
                    device_param_size = total_param_size
                    device_params = params
                fp32_params[device] = (
                    device_params[0].new(0).float().new(device_param_size)
                )
                offset = 0
                for p in device_params:
                    numel = p.data.numel()
                    fp32_params[device][offset : offset + numel].copy_(p.data.view(-1))
                    offset += numel
                fp32_params[device] = torch.nn.Parameter(fp32_params[device])
                fp32_params[device].grad = fp32_params[device].data.new(
                    device_param_size
                )
            return fp32_params
        else:
            fp32_params = []
            for p in params:
                p32 = torch.nn.Parameter(p.data.float())
                p32.grad = torch.zeros_like(p32.data)
                fp32_params.append(p32)
            return fp32_params

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.fp32_optimizer.state_dict()
        if self.scaler is not None:
            state_dict["loss_scale"] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        if "loss_scale" in state_dict and self.scaler is not None:
            self.scaler.loss_scale = state_dict["loss_scale"]
        self.fp32_optimizer.load_state_dict(state_dict, optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        loss.backward()
        self._needs_sync = True

    def _sync_fp16_grads_to_fp32(self):
        if self._needs_sync:
            # copy FP16 grads to FP32
            if self.has_flat_params:
                devices = list(self.fp32_params.keys())
                device_params_dict = defaultdict(list)
                for p in self.fp16_params:
                    if p.requires_grad:
                        device_params_dict[p.device.index].append(p)
                for device in devices:
                    device_params = device_params_dict[device]
                    offset = 0
                    for p in device_params:
                        grad_data = (
                            p.grad.data
                            if p.grad is not None
                            else p.data.new_zeros(p.data.shape)
                        )
                        numel = grad_data.numel()
                        self.fp32_params[device].grad.data[
                            offset : offset + numel
                        ].copy_(grad_data.view(-1))
                        offset += numel
            else:
                for p, p32 in zip(self.fp16_params, self.fp32_params):
                    if not p.requires_grad:
                        continue
                    if p.grad is not None:
                        p32.grad.data.copy_(p.grad.data)
                    else:
                        p32.grad = torch.zeros_like(p.data, dtype=torch.float)

            self._needs_sync = False

    def _sync_fp32_params_to_fp16(self):
        # copy FP32 params back into FP16 model
        if self.has_flat_params:
            devices = list(self.fp32_params.keys())
            device_params_dict = defaultdict(list)
            for p in self.fp16_params:
                device_params_dict[p.device.index].append(p)
            for device in devices:
                device_params = device_params_dict[device]
                offset = 0
                for p in device_params:
                    numel = p.data.numel()
                    p.data.copy_(
                        self.fp32_params[device]
                        .data[offset : offset + numel]
                        .view_as(p.data)
                    )
                    offset += numel
        else:
            for p, p32 in zip(self.fp16_params, self.fp32_params):
                if not p.requires_grad:
                    continue
                p.data.copy_(p32.data)

    def _unscale_grads(self):
        self._sync_fp16_grads_to_fp32()
        if self._multiply_factor != 1.0:
            self.fp32_optimizer.multiply_grads(self._multiply_factor)
            self._multiply_factor = 1.0

    def multiply_grads(self, c):
        """Multiplies grads by a constant ``c``."""
        self._multiply_factor *= c

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm and updates dynamic loss scaler."""
        self._sync_fp16_grads_to_fp32()

        grad_norm = self._multiply_factor * self.fp32_optimizer.clip_grad_norm(
            0, aggregate_norm_fn
        )

        if self.scaler is not None:
            if grad_norm > max_norm > 0.0:
                self._multiply_factor *= max_norm / grad_norm

            self.scaler.check_overflow(grad_norm)
        elif max_norm > 0.0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp_(max=1)
            self._multiply_factor *= clip_coef

        return grad_norm

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._sync_fp16_grads_to_fp32()

        if getattr(self, "supports_step_with_scale", False):
            self.fp32_optimizer.step(closure, scale=(1.0 / self._multiply_factor))
        else:
            self._unscale_grads()
            self.fp32_optimizer.step(closure)

        if self.scaler is not None:
            self.scaler.update()

        self._sync_fp32_params_to_fp16()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.fp16_params:
            p.grad = None
        if self.has_flat_params:
            if torch.is_tensor(self.fp32_params):
                self.fp32_params.grad.zero_()
            elif isinstance(self.fp32_params, dict):
                for fp32_params in self.fp32_params.values():
                    fp32_params.grad.zero_()
            else:
                raise RuntimeError("self.fp32_params must be a tensor or dict")
        else:
            for p32 in self.fp32_params:
                if p32.grad is not None:
                    p32.grad.zero_()
        self._needs_sync = False

        if self.scaler is not None:
            self._multiply_factor = 1.0 / float(self.scaler.loss_scale)


class FP16Optimizer(_FP16OptimizerMixin, optim.FairseqOptimizer):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.
    """

    def __init__(self, cfg: DictConfig, params, fp32_optimizer, fp32_params, **kwargs):
        super().__init__(cfg.optimizer)
        self.fp16_params = params
        self.fp32_optimizer = fp32_optimizer
        self.fp32_params = fp32_params

        if getattr(cfg.common, "fp16_scale_window", None) is None:
            if len(cfg.optimization.update_freq) > 1:
                raise ValueError(
                    "--fp16-scale-window must be given explicitly when using a "
                    "custom --update-freq schedule"
                )
            data_parallel_size = int(
                cfg.distributed_training.distributed_world_size
                / cfg.common.model_parallel_size
            )
            scale_window = int(
                2 ** 14 / data_parallel_size / cfg.optimization.update_freq[0]
            )
        else:
            scale_window = cfg.common.fp16_scale_window

        if not getattr(cfg.common, "bf16", False):
            self.scaler = DynamicLossScaler(
                init_scale=cfg.common.fp16_init_scale,
                scale_window=scale_window,
                tolerance=cfg.common.fp16_scale_tolerance,
                threshold=cfg.common.threshold_loss_scale,
                min_loss_scale=cfg.common.min_loss_scale,
            )
        else:
            # disable loss scaling for bfloat16
            self.scaler = None

    @classmethod
    def build_optimizer(cls, cfg: DictConfig, params, **kwargs):
        """
        Args:
            cfg (omegaconf.DictConfig): fairseq args
            params (iterable): iterable of parameters to optimize
        """
        flatten = not getattr(cfg.common, "fp16_no_flatten_grads", False)
        if getattr(cfg.common, "bf16", False):
            flatten = False  # mixed precision is faster on TPUs without flat grads
        fp32_params = cls.build_fp32_params(cfg.optimizer, params, flatten=flatten)
        if flatten:
            fp32_optimizer = optim.build_optimizer(cfg.optimizer, [fp32_params])
        else:
            fp32_optimizer = optim.build_optimizer(cfg.optimizer, fp32_params)
        if flatten and not fp32_optimizer.supports_flat_params:
            raise RuntimeError(
                f"chosen optimizer {fp32_optimizer.__class__.__name__} does not support flat params, please set --fp16-no-flatten-grads"
            )
        return cls(cfg, params, fp32_optimizer, fp32_params, **kwargs)

    @property
    def optimizer(self):
        return self.fp32_optimizer.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.fp32_optimizer.optimizer = optimizer

    @property
    def optimizer_config(self):
        return self.fp32_optimizer.optimizer_config

    def get_lr(self):
        return self.fp32_optimizer.get_lr()

    def set_lr(self, lr):
        self.fp32_optimizer.set_lr(lr)


class _MemoryEfficientFP16OptimizerMixin(object):
    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in MRO (method resolution order)
        super().__init__(*args, **kwargs)
        self._multiply_factor = 1.0

    @property
    def has_flat_params(self):
        return False

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.wrapped_optimizer.state_dict()
        if self.scaler is not None:
            state_dict["loss_scale"] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        if "loss_scale" in state_dict and self.scaler is not None:
            self.scaler.loss_scale = state_dict["loss_scale"]

        self.wrapped_optimizer.load_state_dict(state_dict, optimizer_overrides)

        # Hack: PyTorch automatically casts the optimizer state to match the
        # type of the current parameters. But with --memory-efficient-fp16 the
        # params are FP16 while the optimizer state is FP32 and we don't want
        # to cast. A workaround is to manually copy back the original state
        # after the optimizer has been loaded.
        if not getattr(self.optimizer, "disable_mem_eff_fp16_loading_hack", False):
            groups = self.optimizer.param_groups
            saved_groups = state_dict["param_groups"]
            id_map = {
                old_id: p
                for old_id, p in zip(
                    chain(*(g["params"] for g in saved_groups)),
                    chain(*(g["params"] for g in groups)),
                )
            }
            for k, v in state_dict["state"].items():
                if k in id_map:
                    param = id_map[k]
                    self.optimizer.state[param] = v

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        loss.backward()

    def _unscale_grads(self):
        if self._multiply_factor != 1.0:
            self.wrapped_optimizer.multiply_grads(self._multiply_factor)
            self._multiply_factor = 1.0

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        self._multiply_factor *= c

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm and updates dynamic loss scaler."""
        max_norm = float(max_norm)
        grad_norm = self._multiply_factor * self.wrapped_optimizer.clip_grad_norm(
            0, aggregate_norm_fn
        )

        if self.scaler is not None:
            grad_norm_cpu = float(grad_norm)
            if grad_norm_cpu > max_norm > 0.0:
                self._multiply_factor *= max_norm / grad_norm_cpu

            # detect overflow and adjust loss scale
            self.scaler.check_overflow(grad_norm_cpu)
        elif max_norm > 0.0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp_(max=1)
            self._multiply_factor *= clip_coef

        return grad_norm

    def step(self, closure=None):
        """Performs a single optimization step."""
        if getattr(self, "supports_step_with_scale", False):
            # NOTE(msb) optimizer divides by scale factor
            self.wrapped_optimizer.step(closure, scale=(1.0 / self._multiply_factor))
        else:
            self._unscale_grads()
            self.wrapped_optimizer.step(closure)

        if self.scaler is not None:
            self.scaler.update()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.wrapped_optimizer.zero_grad()
        if self.scaler is not None:
            self._multiply_factor = 1.0 / float(self.scaler.loss_scale)
        else:
            self._multiply_factor = 1.0


class MemoryEfficientFP16Optimizer(
    _MemoryEfficientFP16OptimizerMixin, optim.FairseqOptimizer
):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.

    Compared to :class:`fairseq.optim.FP16Optimizer`, this version does not
    maintain an FP32 copy of the model. We instead expect the optimizer to
    convert the gradients to FP32 internally and sync the results back to the
    FP16 model params. This significantly reduces memory usage but slightly
    increases the time spent in the optimizer.

    Since this wrapper depends on specific functionality in the wrapped
    optimizer (i.e., on-the-fly conversion of grads to FP32), only certain
    optimizers can be wrapped. This is determined by the
    *supports_memory_efficient_fp16* property.
    """

    def __init__(self, cfg: DictConfig, params, optimizer, **kwargs):
        if not optimizer.supports_memory_efficient_fp16:
            raise ValueError(
                "Unsupported optimizer: {}".format(optimizer.__class__.__name__)
            )

        super().__init__(cfg.optimizer)
        self.wrapped_optimizer = optimizer

        if getattr(cfg.common, "fp16_scale_window", None) is None:
            if len(cfg.optimization.update_freq) > 1:
                raise ValueError(
                    "--fp16-scale-window must be given explicitly when using a "
                    "custom --update-freq schedule"
                )
            data_parallel_size = int(
                cfg.distributed_training.distributed_world_size
                / cfg.common.model_parallel_size
            )
            scale_window = (
                2 ** 14 / data_parallel_size / cfg.optimization.update_freq[0]
            )
        else:
            scale_window = cfg.common.fp16_scale_window

        if not getattr(cfg.common, "bf16", False):
            self.scaler = DynamicLossScaler(
                init_scale=cfg.common.fp16_init_scale,
                scale_window=scale_window,
                tolerance=cfg.common.fp16_scale_tolerance,
                threshold=cfg.common.threshold_loss_scale,
                min_loss_scale=cfg.common.min_loss_scale,
            )
        else:
            # disable loss scaling for bfloat16
            self.scaler = None

    @classmethod
    def build_optimizer(cls, cfg: DictConfig, params, **kwargs):
        """
        Args:
            args (argparse.Namespace): fairseq args
            params (iterable): iterable of parameters to optimize
        """
        fp16_optimizer = optim.build_optimizer(cfg.optimizer, params)
        return cls(cfg, params, fp16_optimizer, **kwargs)

    @property
    def optimizer(self):
        return self.wrapped_optimizer.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.wrapped_optimizer.optimizer = optimizer

    @property
    def optimizer_config(self):
        return self.wrapped_optimizer.optimizer_config

    def get_lr(self):
        return self.wrapped_optimizer.get_lr()

    def set_lr(self, lr):
        self.wrapped_optimizer.set_lr(lr)
