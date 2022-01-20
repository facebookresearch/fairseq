#!/usr/bin/env python3

"""
This module has the EMA class used to store a copy of the exponentially decayed
model params.

Typical usage of EMA class involves initializing an object using an existing
model (random or from a seed model) and setting the config like ema_decay,
ema_start_update which determine how the EMA model is updated. After every
update of the model i.e. at the end of the train_step, the EMA should be updated
by passing the new model to the EMA.step function. The EMA model state dict
can be stored in the extra state under the key of "ema" and dumped
into a checkpoint and loaded. The EMA object can be passed to tasks
by setting task.uses_ema property.
EMA is a smoothed/ensemble model which might have better performance
when used for inference or further fine-tuning. EMA class has a
reverse function to load the EMA params into a model and use it
like a regular model.
"""

import copy
import logging

import torch

from fairseq import checkpoint_utils


class EMA(object):
    """Exponential Moving Average of Fairseq Models
    EMA keeps a copy of the exponentially decayed model params.
    The set of params should include both gradient-descent and
    non-gradient descent params, such as batch mean/var and buffers.
    This is a modified implementation of
    the open source code in https://github.com/zhawe01/fairseq-gec.git,
    and internal source code in
    fbcode/mobile-vision/projects/classification_pytorch/lib/utils/model_ema.py.

    Similar to TF EMA.
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage.
    EMA provides a averaged and smoothed set of model weights, and has been shown to
    improve vision models. EMA class does all necessary functions to update, reload,
    or init EMA methods.

    EMA object is initialized from an arbitrary model. By default, it is stored in
    the same device (unless device specified at initialization) and with the
    same precision as the model (unless ema_fp32 is True). ema_fp32 is recommended.
    This stores the EMA parameters in fp32 only for the EMA update step, and
    is used at the default precision otherwise.
    EMA is usually enabled using EMAConfig with store_ema=True. Some important
    parameters to configure EMA are
    1) ema_decay - The decay of EMA
    2) ema_update_freq - EMA is updated every this many model updates.
    3) ema_start_update - Start EMA update after this many model updates [default 0]

    Key methods:
    1) step - One update of EMA using new model
    2) restore - Update EMA from a state dict
    3) reverse - Load EMA into a model
    4) get_decay, _set_decay - Used to get or set the decay.  Note _set_decay is
    called from step.
    5) build_fp32_params - Used to initialize or update the fp32 copy of EMA params.
    Note this is enabled only when ema_fp32=True
    """

    def __init__(self, model, config, device=None, skip_keys=None):
        """
        @param model model to initialize the EMA with
        @param config EMAConfig object with configuration like
        ema_decay, ema_update_freq, ema_fp32
        @param device If provided, copy EMA to this device (e.g. gpu).
        Otherwise EMA is in the same device as the model.
        """

        self.decay = config.ema_decay
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.config = config
        self.skip_keys = skip_keys or set()
        self.fp32_params = {}

        if self.config.ema_seed_model is not None:
            state = checkpoint_utils.load_ema_from_checkpoint(
                self.config.ema_seed_model
            )
            self.model.load_state_dict(state["model"], strict=True)

        if device is not None:
            logging.info(f"Copying EMA model to device {device}")
            self.model = self.model.to(device=device)

        if self.config.ema_fp32:
            self.build_fp32_params()

        self.update_freq_counter = 0

    def get_model(self):
        return self.model

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
                self.fp32_params[param_key].copy_(state_dict[param_key])
            else:
                self.fp32_params[param_key] = _to_float(state_dict[param_key])

    def restore(self, state_dict, build_fp32_params=False):
        """Load data from a model spec into EMA model"""
        self.model.load_state_dict(state_dict, strict=False)
        if build_fp32_params:
            self.build_fp32_params(state_dict)

    def _set_decay(self, decay):
        self.decay = decay

    def get_decay(self):
        return self.decay

    def _step_internal(self, new_model, updates=None):
        """One update of the EMA model based on new model weights"""
        decay = self.decay

        ema_state_dict = {}
        ema_params = (
            self.fp32_params if self.config.ema_fp32 else self.model.state_dict()
        )
        for key, param in new_model.state_dict().items():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = (
                    param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                )

            if param.shape != ema_param.shape:
                raise ValueError(
                    "incompatible tensor shapes between model param and ema param"
                    + "{} vs. {}".format(param.shape, ema_param.shape)
                )

            if "version" in key:
                # Do not decay a model.version pytorch param
                continue

            if key in self.skip_keys:
                ema_param = param.to(dtype=ema_param.dtype).clone()
            else:
                ema_param.mul_(decay)
                ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - decay)
            ema_state_dict[key] = ema_param
        self.restore(ema_state_dict, build_fp32_params=False)

    def step(self, new_model, updates=None):
        """
        One update of EMA which is done every self.config.ema_update_freq
        updates of the model.

        @param updates The current number of model updates done.
        Decay is set of 0 if model updates < ema_start_update, which means
        the model will be simply copied over to the EMA.
        When model updates >= ema_start_updates, then EMA is updated with
        a decay of self.config.ema_decay.
        """
        if updates is not None:
            self._set_decay(
                0 if updates < self.config.ema_start_update else self.config.ema_decay
            )
            if updates is not None and self.config.ema_update_freq > 1:
                self.update_freq_counter += 1
                if self.update_freq_counter >= self.config.ema_update_freq:
                    self._step_internal(new_model, updates)
                    self.update_freq_counter = 0
        else:
            self._step_internal(new_model, updates)

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
