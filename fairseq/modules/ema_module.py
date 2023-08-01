#!/usr/bin/env python3

"""
Used for EMA tracking a given pytorch module. The user is responsible for calling step()
and setting the appropriate decay
"""

import copy
from dataclasses import dataclass, field
import logging

import torch

from omegaconf import II
from fairseq.dataclass import FairseqDataclass

try:
    from amp_C import multi_tensor_l2norm

    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False

logger = logging.getLogger(__name__)


@dataclass
class EMAModuleConfig(FairseqDataclass):
    ema_decay: float = field(
        default=0.9999, metadata={"help": "decay for exponential moving average model"}
    )
    ema_fp32: bool = field(
        default=False,
        metadata={"help": "If true, store EMA model in fp32 even if model is in fp16"},
    )
    add_missing_params: bool = True
    log_norms: bool = False


class EMAModule:
    """Exponential Moving Average of Fairseq Models"""

    def __init__(
        self,
        model,
        config: EMAModuleConfig,
        copy_model=True,
        device=None,
        skip_keys=None,
    ):
        """
        @param model model to initialize the EMA with
        @param config EMAConfig object with configuration like
        ema_decay, ema_update_freq, ema_fp32
        @param device If provided, copy EMA to this device (e.g. gpu).
        Otherwise EMA is in the same device as the model.
        """

        self.config = config

        if copy_model:
            self.model = copy.deepcopy(model)
            self.model.requires_grad_(False)
        else:
            self.model = model

        self.config = config
        self.decay = config.ema_decay
        self.skip_keys = skip_keys or set()
        self.add_missing_params = config.add_missing_params
        self.fp32_params = {}

        if device is not None:
            logging.info(f"Copying EMA model to device {device}")
            self.model = self.model.to(device=device)

        if self.config.ema_fp32:
            self.build_fp32_params()

        self.log_norms = config.log_norms and multi_tensor_l2norm_available
        self.logs = {}

    def build_fp32_params(self, state_dict=None):
        """
        Store a copy of the EMA params in fp32.
        If state dict is passed, the EMA params is copied from
        the provided state dict. Otherwise, it is copied from the
        current EMA model parameters.
        """
        if not self.config.ema_fp32:
            raise RuntimeError(
                "build_fp32_params should not be called if ema_fp32=False. "
                "Use ema_fp32=True if this is really intended."
            )

        if state_dict is None:
            state_dict = self.model.state_dict()

        def _to_float(t):
            return t.float() if torch.is_floating_point(t) else t

        for param_key in state_dict:
            if param_key in self.fp32_params:
                if param_key == "__sq_mom":
                    self.fp32_params[param_key] = state_dict[param_key]
                else:
                    self.fp32_params[param_key].copy_(state_dict[param_key])
            else:
                self.fp32_params[param_key] = _to_float(state_dict[param_key])
                if "__sq_mom" in self.fp32_params:
                    self.fp32_params["__sq_mom"][param_key] = torch.zeros_like(
                        self.fp32_params[param_key]
                    )

    def restore(self, state_dict, build_fp32_params=False):
        """Load data from a model spec into EMA model"""
        self.model.load_state_dict(state_dict, strict=False)
        if build_fp32_params:
            self.build_fp32_params(state_dict)

    def set_decay(self, decay, weight_decay=None):
        self.decay = decay
        if weight_decay is not None:
            self.weight_decay = weight_decay

    def get_decay(self):
        return self.decay

    def _step_internal(self, new_model):
        """One update of the EMA model based on new model weights"""
        decay = self.decay

        ema_state_dict = {}
        ema_params = (
            self.fp32_params if self.config.ema_fp32 else self.model.state_dict()
        )

        new_p = []
        ema_p = []

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue

            if not self.add_missing_params and key not in ema_params:
                continue

            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = (
                    param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                )
                ema_params[key] = ema_param

            if param.shape != ema_param.shape:
                raise ValueError(
                    "incompatible tensor shapes between model param and ema param"
                    + "{} vs. {}".format(param.shape, ema_param.shape)
                )

            if "version" in key:
                # Do not decay a model.version pytorch param
                continue

            lr = 1 - decay

            if key in self.skip_keys or not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                if self.log_norms:
                    new_p.append(param)
                    ema_p.append(ema_param)

                ema_param.mul_(1 - lr)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=lr)

            ema_state_dict[key] = ema_param

        for key, param in new_model.named_buffers():
            ema_state_dict[key] = param

        if self.log_norms:
            if "model_norm" in self.logs:
                self.prev_model_norm = self.logs["model_norm"]

            chunk_size = 2048 * 32
            has_inf = torch.zeros(
                (1, 1), dtype=torch.int, device=next(new_model.parameters()).device
            )

            new_norm = multi_tensor_l2norm(chunk_size, has_inf, [new_p], False)
            old_norm = multi_tensor_l2norm(chunk_size, has_inf, [ema_p], False)

            self.logs["model_norm"] = new_norm[0]
            self.logs["ema_norm"] = old_norm[0]

        self.restore(ema_state_dict, build_fp32_params=False)

    @torch.no_grad()
    def step(self, new_model):
        self._step_internal(new_model)

    def reverse(self, model):
        """
        Load the model parameters from EMA model.
        Useful for inference or fine-tuning from the EMA model.
        """
        d = self.model.state_dict()
        if "_ema" in d:
            del d["_ema"]

        model.load_state_dict(d, strict=False)
        return model
