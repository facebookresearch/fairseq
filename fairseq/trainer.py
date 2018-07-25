# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Train a network across multiple GPUs.
"""

from collections import defaultdict, OrderedDict
import contextlib
from itertools import chain

import torch

from fairseq import distributed_utils, optim, utils
from fairseq.meters import AverageMeter, TimeMeter
from fairseq.optim import lr_scheduler


class Trainer(object):
    """Main class for data parallel training.

    This class supports data parallel training, where multiple workers each
    have a full model replica and gradients are accumulated synchronously via
    torch.distributed.all_reduce.
    """

    def __init__(self, args, task, model, criterion):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.args = args

        # copy model and criterion to current device
        self.task = task
        self.model = model.cuda()
        self.criterion = criterion.cuda()

        # initialize meters
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        self.meters['wall'] = TimeMeter()      # wall time in seconds

        self._buffered_stats = defaultdict(lambda: [])
        self._flat_grads = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    def _build_optimizer(self):
        self._optimizer = optim.build_optimizer(self.args, self.model.parameters())
        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self._optimizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state['train_meters'] = self.meters
            utils.save_state(
                filename, self.args, self.model, self.criterion, self.optimizer,
                self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
            )

    def load_checkpoint(self, filename):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_model_state(filename, self.model)

        if last_optim_state is not None:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            if last_optim['criterion_name'] == self.criterion.__class__.__name__:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                if last_optim['optimizer_name'] == self.optimizer.__class__.__name__:
                    self.optimizer.load_state_dict(last_optim_state)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters = extra_state['train_meters']
            del extra_state['train_meters']

        return extra_state

    def train_step(self, sample, update_params=True):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # forward and backward pass
        sample = self._prepare_sample(sample)
        loss, sample_size, logging_output, oom_fwd = self._forward(sample)
        oom_bwd = self._backward(loss)

        # buffer stats and logging outputs
        self._buffered_stats['sample_sizes'].append(sample_size)
        self._buffered_stats['logging_outputs'].append(logging_output)
        self._buffered_stats['ooms_fwd'].append(oom_fwd)
        self._buffered_stats['ooms_bwd'].append(oom_bwd)

        # update parameters
        if update_params:
            # gather logging outputs from all replicas
            sample_sizes = self._buffered_stats['sample_sizes']
            logging_outputs = self._buffered_stats['logging_outputs']
            ooms_fwd = self._buffered_stats['ooms_fwd']
            ooms_bwd = self._buffered_stats['ooms_bwd']
            if self.args.distributed_world_size > 1:
                sample_sizes, logging_outputs, ooms_fwd, ooms_bwd = map(
                    lambda l: list(chain.from_iterable(l)),
                    zip(*distributed_utils.all_gather_list(
                        (sample_sizes, logging_outputs, ooms_fwd, ooms_bwd)
                    ))
                )
            ooms_fwd = sum(ooms_fwd)
            ooms_bwd = sum(ooms_bwd)

            if ooms_fwd == self.args.distributed_world_size:
                print('| WARNING: OOM in all workers, skipping batch')
                self.zero_grad()
                return None

            # aggregate stats and logging outputs
            ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
            nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
            agg_logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)
            grad_denom = self.criterion.__class__.grad_denom(sample_sizes)

            try:
                # all-reduce and rescale gradients, then take an optimization step
                grad_norm = self._all_reduce_and_rescale(grad_denom)
                self._opt()

                # update meters
                self.meters['wps'].update(ntokens)
                self.meters['ups'].update(1.)
                self.meters['wpb'].update(ntokens)
                self.meters['bsz'].update(nsentences)
                if grad_norm is not None:
                    self.meters['gnorm'].update(grad_norm)
                    self.meters['clip'].update(1. if grad_norm > self.args.clip_norm else 0.)
                self.meters['oom'].update(ooms_fwd + ooms_bwd)

                # update loss meters for training
                if 'loss' in agg_logging_output:
                    self.meters['train_loss'].update(agg_logging_output['loss'], grad_denom)
                # criterions can optionally log the NLL loss too
                if 'nll_loss' in agg_logging_output:
                    self.meters['train_nll_loss'].update(agg_logging_output['nll_loss'], ntokens)
            except OverflowError as e:
                self.zero_grad()
                print('| WARNING: overflow detected, ' + str(e))

            self.clear_buffered_stats()

            return agg_logging_output
        else:
            return None  # buffering updates

    def _forward(self, sample, eval=False):
        loss = None
        sample_size = 0
        logging_output = {
            'ntokens': sample['ntokens'] if sample is not None else 0,
            'nsentences': sample['target'].size(0) if sample is not None else 0,
        }
        oom = 0
        try:
            # prepare model and optimizer
            if eval:
                self.model.eval()
            else:
                self.model.train()
                self.optimizer.zero_grad()

            if sample is not None:
                with torch.no_grad() if eval else contextlib.ExitStack():
                    # calculate loss and sample size
                    loss, sample_size, logging_output_ = self.task.get_loss(self.model, self.criterion, sample)
                    logging_output.update(logging_output_)
        except RuntimeError as e:
            if not eval and 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                oom = 1
                loss = None
            else:
                raise e
        return loss, sample_size, logging_output, oom

    def _backward(self, loss):
        oom = 0
        if loss is not None:
            try:
                # backward pass
                loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    self.zero_grad()
                else:
                    raise e
        return oom

    def _all_reduce_and_rescale(self, grad_denom):
        # flatten grads into a single buffer and all-reduce
        flat_grads = self._flat_grads = self._get_flat_grads(self._flat_grads)
        if self.args.distributed_world_size > 1:
            torch.distributed.all_reduce(flat_grads)

        # rescale and clip gradients
        flat_grads.div_(grad_denom)
        grad_norm = utils.clip_grad_norm_(flat_grads, self.args.clip_norm)

        # copy grads back into model parameters
        self._set_flat_grads(flat_grads)

        return grad_norm

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                   'Use the param in the forward pass or set requires_grad=False')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset+numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def _set_flat_grads(self, new_grads):
        grads = self._get_grads()
        offset = 0
        for g in grads:
            numel = g.numel()
            g.copy_(new_grads[offset:offset+numel].view_as(g))
            offset += numel

    def _opt(self):
        # take an optimization step
        self.optimizer.step()
        self.zero_grad()
        self._num_updates += 1

        # update learning rate
        self.lr_scheduler.step_update(self._num_updates)

    def valid_step(self, sample):
        """Do forward pass in evaluation mode."""
        # forward pass
        sample = self._prepare_sample(sample)
        _loss, sample_size, logging_output, oom_fwd = self._forward(sample, eval=True)
        assert not oom_fwd, 'Ran out of memory during validation'

        # gather logging outputs from all GPUs
        if self.args.distributed_world_size > 1:
            sample_sizes, logging_outputs = zip(*distributed_utils.all_gather_list(
                (sample_size, logging_output)
            ))
        else:
            sample_sizes = [sample_size]
            logging_outputs = [logging_output]

        # aggregate stats and logging outputs
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        grad_denom = self.criterion.__class__.grad_denom(sample_sizes)
        agg_logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)

        # update loss meters for validation
        if 'loss' in agg_logging_output:
            self.meters['valid_loss'].update(agg_logging_output['loss'], grad_denom)
        # criterions can optionally log the NLL loss too
        if 'nll_loss' in agg_logging_output:
            self.meters['valid_nll_loss'].update(agg_logging_output['nll_loss'], ntokens)

        return agg_logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, update_params=False)
        self.zero_grad()
        self.clear_buffered_stats()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clear_buffered_stats(self):
        self._buffered_stats.clear()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the model replica."""
        return self.model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)
