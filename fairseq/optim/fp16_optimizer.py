# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain

import torch

from fairseq import optim, utils


class DynamicLossScaler(object):

    def __init__(
        self, init_scale=2.**15, scale_factor=2., scale_window=2000,
        tolerance=0.05, threshold=None,
    ):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold
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
                self._decrease_loss_scale()
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
        self._iter += 1

    def _decrease_loss_scale(self):
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)

    @staticmethod
    def has_overflow(grad_norm):
        # detect inf and nan
        if grad_norm == float('inf') or grad_norm != grad_norm:
            return True
        return False


class _FP16OptimizerMixin(object):

    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)

    @classmethod
    def build_fp32_params(cls, params):
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
        return fp32_params

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
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        loss = loss * self.scaler.loss_scale
        loss.backward()
        self._needs_sync = True

    def _sync_fp16_grads_to_fp32(self, multiply_grads=1.):
        if self._needs_sync:
            # copy FP16 grads to FP32
            offset = 0
            for p in self.fp16_params:
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
            if self.scaler.loss_scale <= self.min_loss_scale:
                # Use FloatingPointError as an uncommon error that parent
                # functions can safely catch to stop training.
                raise FloatingPointError((
                    'Minimum loss scale reached ({}). Your loss is probably exploding. '
                    'Try lowering the learning rate, using gradient clipping or '
                    'increasing the batch size.'
                ).format(self.min_loss_scale))
            raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))
        return grad_norm

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._sync_fp16_grads_to_fp32()
        self.fp32_optimizer.step(closure)

        # copy FP32 params back into FP16 model
        offset = 0
        for p in self.fp16_params:
            if not p.requires_grad:
                continue
            numel = p.data.numel()
            p.data.copy_(self.fp32_params.data[offset:offset+numel].view_as(p.data))
            offset += numel

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.fp16_params:
            p.grad = None
        self._needs_sync = False


class FP16Optimizer(_FP16OptimizerMixin, optim.FairseqOptimizer):
    """
    Wrap an *optimizer* to support FP16 (mixed precision) training.
    """

    def __init__(self, args, params, fp32_optimizer, fp32_params):
        super().__init__(args)
        self.fp16_params = params
        self.fp32_optimizer = fp32_optimizer
        self.fp32_params = fp32_params

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
            threshold=args.threshold_loss_scale,
        )
        self.min_loss_scale = self.args.min_loss_scale

    @classmethod
    def build_optimizer(cls, args, params):
        """
        Args:
            args (argparse.Namespace): fairseq args
            params (iterable): iterable of parameters to optimize
        """
        fp32_params = cls.build_fp32_params(params)
        fp32_optimizer = optim.build_optimizer(args, [fp32_params])
        return cls(args, params, fp32_optimizer, fp32_params)

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


class _MemoryEfficientFP16OptimizerMixin(object):

    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)

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

        self.wrapped_optimizer.load_state_dict(state_dict, optimizer_overrides)

        # Hack: PyTorch automatically casts the optimizer state to match the
        # type of the current parameters. But with --memory-efficient-fp16 the
        # params are FP16 while the optimizer state is FP32 and we don't want
        # to cast. A workaround is to manually copy back the original state
        # after the optimizer has been loaded.
        groups = self.optimizer.param_groups
        saved_groups = state_dict['param_groups']
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain(*(g['params'] for g in saved_groups)),
                chain(*(g['params'] for g in groups))
            )
        }
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                self.optimizer.state[param] = v

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
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
        """Multiplies grads by a constant *c*."""
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
            if self.scaler.loss_scale <= self.min_loss_scale:
                # Use FloatingPointError as an uncommon error that parent
                # functions can safely catch to stop training.
                raise FloatingPointError((
                    'Minimum loss scale reached ({}). Your loss is probably exploding. '
                    'Try lowering the learning rate, using gradient clipping or '
                    'increasing the batch size.'
                ).format(self.min_loss_scale))
            raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))

        return grad_norm

    def step(self, closure=None):
        """Performs a single optimization step."""
        self._unscale_grads()
        self.wrapped_optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.wrapped_optimizer.zero_grad()
        self._grads_are_scaled = False


class MemoryEfficientFP16Optimizer(_MemoryEfficientFP16OptimizerMixin, optim.FairseqOptimizer):
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

    def __init__(self, args, params, optimizer):
        if not optimizer.supports_memory_efficient_fp16:
            raise ValueError(
                'Unsupported optimizer: {}'.format(optimizer.__class__.__name__)
            )

        super().__init__(args)
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
            threshold=args.threshold_loss_scale,
        )
        self.min_loss_scale = self.args.min_loss_scale

    @classmethod
    def build_optimizer(cls, args, params):
        """
        Args:
            args (argparse.Namespace): fairseq args
            params (iterable): iterable of parameters to optimize
        """
        fp16_optimizer = optim.build_optimizer(args, params)
        return cls(args, params, fp16_optimizer)

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
