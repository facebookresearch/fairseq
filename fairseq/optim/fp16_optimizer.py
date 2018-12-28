# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import optim, utils


class DynamicLossScaler(object):

    def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000, tolerance=0.05):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0

    def update_scale(self, overflow):
        iter_since_rescale = self._iter - self._last_rescale_iter
        if overflow:
            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                self.loss_scale /= self.scale_factor
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
        self._iter += 1

    @staticmethod
    def has_overflow(grad_norm):
        # detect inf and nan
        if grad_norm == float('inf') or grad_norm != grad_norm:
            return True
        return False


class ConvertToFP32(object):
    """
    A wrapper around a list of params that will convert them to FP32 on the
    first iteration, after which this essentially behaves like a normal list.
    """

    def __init__(self, params):

        def convert_to_fp32(p):
            p.data = p.data.float()
            if p.grad is not None:
                p.grad.data = p.grad.data.float()
            return p

        assert isinstance(params, list)
        self.params = params
        self.itr = map(convert_to_fp32, params)

    @staticmethod
    def wrap_optimizer_(optimizer):
        for group in optimizer.param_groups:
            group['params'] = ConvertToFP32(group['params'])

    @staticmethod
    def unwrap_optimizer_(optimizer):
        for group in optimizer.param_groups:
            group['params'] = group['params'].params  # unwrap from ConvertToFP32
            for p in group['params']:
                p.data = p.data.half()
                if p.grad is not None:
                    p.grad.data = p.grad.data.half()

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        if self.itr is not None:
            return self
        else:
            return iter(self.params)

    def __next__(self):
        try:
            return next(self.itr)
        except StopIteration:
            self.itr = None
            raise StopIteration


class FP16Optimizer(optim.FairseqOptimizer):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.

    Args:
        args (argparse.Namespace): fairseq args
        params (iterable): iterable of parameters to optimize
        optimizer (~fairseq.optim.FairseqOptimizer): optimizer to wrap
    """

    def __init__(self, args, params, optimizer):
        super().__init__(args, params)
        self.wrapped_optimizer = optimizer

        if getattr(args, 'fp16_scale_window', None) is None:
            if len(args.update_freq) > 1:
                raise ValueError(
                    '--fp16-scale-window must be given explicitly when using a '
                    'custom --update-freq schedule'
                )
            scale_window = 2**14 / args.distributed_world_size / args.update_freq[0]
        else:
            scale_window = args.fp16_scale_window

        self.scaler = DynamicLossScaler(
            init_scale=args.fp16_init_scale,
            scale_window=scale_window,
            tolerance=args.fp16_scale_tolerance,
        )

    @staticmethod
    def build_optimizer(args, params):
        fp16_optimizer = optim.build_optimizer(args, params)
        return FP16Optimizer(args, params, fp16_optimizer)

    @property
    def optimizer(self):
        return self.wrapped_optimizer.optimizer

    @property
    def optimizer_config(self):
        return self.wrapped_optimizer.optimizer_config

    def get_lr(self):
        return self.wrapped_optimizer.get_lr()

    def set_lr(self, lr):
        self.wrapped_optimizer.set_lr(lr)

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.wrapped_optimizer.state_dict()
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
        ConvertToFP32.wrap_optimizer_(self.wrapped_optimizer.optimizer)
        self.wrapped_optimizer.load_state_dict(state_dict, optimizer_overrides)
        ConvertToFP32.unwrap_optimizer_(self.wrapped_optimizer.optimizer)

    def backward(self, loss):
        loss = loss * self.scaler.loss_scale
        loss.backward()
        self._grads_are_scaled = True

    def _unscale_grads(self, multiply_grads=1.):
        if self._grads_are_scaled:
            self._grads_are_scaled = False

            # correct for dynamic loss scaler
            self.wrapped_optimizer.multiply_grads(multiply_grads / self.scaler.loss_scale)
        else:
            assert multiply_grads == 1.

    def multiply_grads(self, c):
        """Multiplies grads by a constant ``c``."""
        if self._grads_are_scaled:
            self._unscale_grads(c)
        else:
            self.wrapped_optimizer.multiply_grads(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm and updates dynamic loss scaler."""
        self._unscale_grads()
        grad_norm = self.wrapped_optimizer.clip_grad_norm(max_norm)

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
        self._unscale_grads()

        # convert params and grads to FP32 (lazily)
        ConvertToFP32.wrap_optimizer_(self.wrapped_optimizer.optimizer)

        self.wrapped_optimizer.step(closure)

        # convert params back to FP16
        ConvertToFP32.unwrap_optimizer_(self.wrapped_optimizer.optimizer)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.wrapped_optimizer.zero_grad()
        self._grads_are_scaled = False
