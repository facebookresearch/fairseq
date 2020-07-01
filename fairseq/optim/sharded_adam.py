# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
import torch.optim

from fairseq import distributed_utils
from fairseq.optim import FairseqOptimizer, register_optimizer

logger = logging.getLogger(__name__)


@register_optimizer('sharded_adam')
class FairseqShardedAdam(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Compared to FairseqAdam, this optimizer will shard the optimizer state
    across data parallel replicas (i.e., "Optimizer State Sharding" from the
    ZeRO paper). It is not permitted to use the `--fp16` or `--bf16` options,
    since this optimizer shards the FP32 copy of the model internally and thus
    using the `--memory-efficient-fp16` or `--memory-efficient-bf16` options
    will provide identical results while being more efficient.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        if (
            (getattr(args, 'fp16', False) or getattr(args, 'bf16', False))
            and not (
                getattr(args, 'memory_efficient_fp16', False)
                or getattr(args, 'memory_efficient_bf16', False)
            )
        ):
            raise ValueError(
                '--optimizer=sharded_adam is not compatible with --fp16 or --bf16; '
                'use --memory-efficient-fp16 or --memory-efficient-bf16 instead'
            )
        self._optimizer = ShardedAdam(
            params,
            process_group=distributed_utils.get_data_parallel_group(),
            **self.optimizer_config,
        )
        self._multiply_factor = 1.

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
        }

    def all_reduce_grads(self, module):
        """Manually all-reduce gradients (if required)."""
        if not hasattr(module, "all_reduce_grads"):
            raise RuntimeError(
                "--optimizer=sharded_adam requires a DDP wrapper that manually "
                "reduces gradients, such as --ddp-backend=no_c10d or --tpu"
            )

        # we don't reduce grads here, since ShardedAdam will reduce the grads itself
        #module.all_reduce_grads()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        self._multiply_factor *= c

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""

        def aggregated_sharded_grad_norm(total_norm):
            if aggregate_norm_fn is not None:
                total_norm = aggregate_norm_fn(total_norm)
            # report the average norm across data parallel workers
            distributed_utils.all_reduce(
                total_norm, group=distributed_utils.get_data_parallel_group()
            )
            return total_norm / distributed_utils.get_data_parallel_world_size()

        # don't actually clip norm now, just record the norm and update
        # self._multiply_factor
        grad_norm = super().clip_grad_norm(
            max_norm=0, aggregate_norm_fn=aggregated_sharded_grad_norm
        )
        if max_norm > 0.0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp_(max=1)
            self._multiply_factor *= clip_coef
        return grad_norm

    def step(self, closure=None, scale=1.):
        """Performs a single optimization step."""
        # optimizers divide by scale, so we divide by multiply_factor
        scale /= self._multiply_factor
        return super().step(closure=closure, scale=scale)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self._multiply_factor = 1.
        super().zero_grad()


class ShardedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    This implementation is modified from torch.optim.Adam based on:
    `Fixed Weight Decay Regularization in Adam`
    (see https://arxiv.org/abs/1711.05101)

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        process_group,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ShardedAdam, self).__init__(params, defaults)

        self.process_group = process_group
        self.shard_id = distributed_utils.get_rank(self.process_group)
        self.num_shards = distributed_utils.get_world_size(self.process_group)

        for p in params:
            assert p.numel() % self.num_shards == 0, (
                'param with size {} not divisible by number of shards ({})'
                .format(p.size(), self.num_shards)
            )

        self._first_run = True

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    @property
    def supports_step_with_scale(self):
        return True

    @torch.no_grad()
    def step(self, closure=None, scale=1.):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('ShardedAdam does not support sparse gradients')

                part_size = grad.numel() // self.num_shards
                s = self.shard_id * part_size
                e = (self.shard_id + 1) * part_size

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    ref = p.view(-1)[s:e].float()
                    if p.dtype in {torch.float16, torch.bfloat16}:
                        # FP32 model weights
                        state['fp32_weight'] = ref
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(ref)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(ref)
                elif self._first_run:
                    ref = p.view(-1)[:0].float()  # just store type/device info
                    if 'fp32_weight' in state:
                        state['fp32_weight'] = state['fp32_weight'].to(ref)
                    state['exp_avg'] = state['exp_avg'].to(ref)
                    state['exp_avg_sq'] = state['exp_avg_sq'].to(ref)
                    self._first_run = False

                # call all_to_all on grad.view(-1)
                grad = distributed_utils.all_to_all(
                    tensor=grad.view(-1),
                    group=self.process_group,
                )

                # average grads across data parallel workers
                grad = grad.view(self.num_shards, part_size).mean(dim=0)

                # cast to FP32 before normalizing by scale
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                # normalize grad by scale
                grad.div_(scale)

                if 'fp32_weight' in state:
                    p_fp32_part = state['fp32_weight']
                else:
                    p_fp32_part = p.view(-1)[s:e]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['weight_decay'] != 0:
                    p_fp32_part.add_(
                        p_fp32_part, alpha=-group['weight_decay'] * group['lr']
                    )

                p_fp32_part.addcdiv_(exp_avg, denom, value=-step_size)

                # possibly convert back to FP16/BF16
                p_part = p_fp32_part.to(p)

                # gather the updated parameter
                p_complete = distributed_utils.all_gather(
                    tensor=p_part,
                    group=self.process_group,
                    return_tensor=True,
                )
                p.copy_(p_complete.view_as(p))

        return loss
