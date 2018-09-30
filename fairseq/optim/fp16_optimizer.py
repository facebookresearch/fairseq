# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import optim, utils


class DynamicLossScaler:

    def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self._iter = 0
        self._last_overflow_iter = -1

    def update_scale(self, overflow):
        if overflow:
            self.loss_scale /= self.scale_factor
            self._last_overflow_iter = self._iter
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
        self._iter += 1

    @staticmethod
    def has_overflow(grad_norm):
        # detect inf and nan
        if grad_norm == float('inf') or grad_norm != grad_norm:
            return True
        return False


class FP16Optimizer(optim.FairseqOptimizer):

    def __init__(self, args, params, fp32_optimizer, fp32_params):
        super().__init__(args, params)
        self.fp32_optimizer = fp32_optimizer
        self.fp32_params = fp32_params
        self.scaler = DynamicLossScaler(
            init_scale=args.fp16_init_scale,
            scale_window=(2**14 / args.distributed_world_size),
        )

    @staticmethod
    def build_optimizer(args, params):
        # create FP32 copy of parameters and grads
        total_param_size = sum(p.data.numel() for p in params)
        fp32_params = params[0].new(0).float().new(total_param_size)
        offset = 0
        for p in params:
            numel = p.data.numel()
            fp32_params[offset:offset+numel].copy_(p.data.view(-1))
            offset += numel
        fp32_params = torch.nn.Parameter(fp32_params)
        fp32_params.grad = fp32_params.data.new(total_param_size)

        fp32_optimizer = optim.build_optimizer(args, [fp32_params])
        return FP16Optimizer(args, params, fp32_optimizer, fp32_params)

    @property
    def optimizer(self):
        return self.fp32_optimizer.optimizer

    @property
    def optimizer_config(self):
        return self.fp32_optimizer.optimizer_config

    def get_lr(self):
        return self.fp32_optimizer.get_lr()

    def set_lr(self, lr):
        self.fp32_optimizer.set_lr(lr)

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.fp32_optimizer.state_dict()
        state_dict['loss_scale'] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        if 'loss_scale' in state_dict:
            self.scaler.loss_scale = state_dict['loss_scale']
        self.fp32_optimizer.load_state_dict(state_dict, optimizer_overrides)

    def backward(self, loss):
        loss = loss * self.scaler.loss_scale
        loss.backward()
        self._needs_sync = True

    def _sync_fp16_grads_to_fp32(self, multiply_grads=1.):
        if self._needs_sync:
            # copy FP16 grads to FP32
            offset = 0
            for p in self.params:
                if not p.requires_grad:
                    continue
                grad_data = p.grad.data if p.grad is not None else p.data.new_zeros(p.data.shape)
                numel = grad_data.numel()
                self.fp32_params.grad.data[offset:offset+numel].copy_(grad_data.view(-1))
                offset += numel

            # correct for dynamic loss scaler
            self.fp32_params.grad.data.mul_(multiply_grads / self.scaler.loss_scale)

            self._needs_sync = False

    def multiply_grads(self, c):
        """Multiplies grads by a constant ``c``."""
        if self._needs_sync:
            self._sync_fp16_grads_to_fp32(c)
        else:
            self.fp32_params.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm and updates dynamic loss scaler."""
        self._sync_fp16_grads_to_fp32()
        grad_norm = utils.clip_grad_norm_(self.fp32_params.grad.data, max_norm)

        # detect overflow and adjust loss scale
        overflow = DynamicLossScaler.has_overflow(grad_norm)
        self.scaler.update_scale(overflow)
        if overflow:
            if self.scaler.loss_scale <= self.args.min_loss_scale:
                # Use FloatingPointError as an uncommon error that parent
                # functions can safely catch to stop training.
                raise FloatingPointError((
                    'Minimum loss scale reached ({}). Your loss is probably exploding. '
                    'Try lowering the learning rate, using gradient clipping or '
                    'increasing the batch size.'
                ).format(self.args.min_loss_scale))
            raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))
        return grad_norm

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._sync_fp16_grads_to_fp32()
        self.fp32_optimizer.step(closure)

        # copy FP32 params back into FP16 model
        offset = 0
        for p in self.params:
            if not p.requires_grad:
                continue
            numel = p.data.numel()
            p.data.copy_(self.fp32_params.data[offset:offset+numel].view_as(p.data))
            offset += numel

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.fp32_optimizer.zero_grad()
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        self._needs_sync = False
