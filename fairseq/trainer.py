# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Train a network on multiple GPUs.
"""

from collections import OrderedDict
import math
import torch

from fairseq import distributed_utils, optim, utils
from fairseq.meters import AverageMeter, TimeMeter
from fairseq.optim import lr_scheduler


class Trainer(object):
    """Main class for multi-GPU training.

    Each GPU has a full copy of the model and is assigned to its own Python
    process. Gradients are accumulated with torch.distributed.all_reduce and all
    model replicas are updated synchronously after each batch.
    """

    def __init__(self, args, model, criterion):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.args = args

        # copy model and criterion to current device
        self.model = model.cuda()
        self.criterion = criterion.cuda()

        # initialize optimizer and LR scheduler
        self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

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

        self._max_bsz_seen = 0
        self._num_updates = 0
        self._optim_history = None

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            utils.save_state(filename, self.args, self.model, self.criterion, self.optimizer,
                             self.lr_scheduler, self._num_updates, self._optim_history, extra_state)

    def load_checkpoint(self, filename):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = utils.load_model_state(
            filename, self.model, cuda_device=torch.cuda.current_device())

        if last_optim_state is not None:
            # rebuild optimizer after loading model, since params may have changed
            self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
            self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            if last_optim['criterion_name'] == self.criterion.__class__.__name__:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                if last_optim['optimizer_name'] == self.optimizer.__class__.__name__:
                    self.optimizer.load_state_dict(last_optim_state)

            self._num_updates = last_optim['num_updates']

        return extra_state

    def train_step(self, sample):
        """Do forward, backward and parameter update."""

        sample = self._prepare_sample(sample, volatile=False)

        # forward pass
        loss, sample_sizes, logging_outputs, ooms_fwd = self._forward(sample)

        # aggregate stats and logging outputs
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        grad_denom = self.criterion.__class__.grad_denom(sample_sizes)
        agg_logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)

        # backward pass, all-reduce gradients and take an optimization step
        grad_norm, ooms_bwd = self._backward_and_opt(loss, grad_denom)

        # update meters
        self.meters['wps'].update(ntokens)
        self.meters['ups'].update(1.)
        self.meters['wpb'].update(ntokens)
        self.meters['bsz'].update(nsentences)
        self.meters['gnorm'].update(grad_norm)
        self.meters['clip'].update(1. if grad_norm > self.args.clip_norm else 0.)
        self.meters['oom'].update(ooms_fwd + ooms_bwd)

        # update loss meters for training
        if 'loss' in agg_logging_output:
            self.meters['train_loss'].update(agg_logging_output['loss'], grad_denom)
        # criterions can optionally log the NLL loss too
        if 'nll_loss' in agg_logging_output:
            self.meters['train_nll_loss'].update(agg_logging_output['nll_loss'], ntokens)

        return agg_logging_output

    def _forward(self, sample, eval=False):
        # prepare model and optimizer
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        loss = None
        sample_size = 0
        logging_output = {
            'ntokens': sample['ntokens'] if sample is not None else 0,
            'nsentences': sample['target'].size(0) if sample is not None else 0,
        }
        oom = 0
        if sample is not None:
            try:
                with utils.maybe_no_grad(eval):
                    # calculate loss and sample size
                    loss, sample_size, logging_output_ = self.criterion(self.model, sample)
                    logging_output.update(logging_output_)
            except RuntimeError as e:
                if not eval and 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    loss = None
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

        # synchronize logging outputs for multi-GPU training
        if self.args.distributed_world_size > 1:
            sample_sizes, logging_outputs, ooms = zip(*list(
                distributed_utils.all_gather_list((sample_size, logging_output, oom))))
            ooms = sum(ooms)
        else:
            sample_sizes = [sample_size]
            logging_outputs = [logging_output]
            ooms = oom

        return loss, sample_sizes, logging_outputs, ooms

    def _backward_and_opt(self, loss, grad_denom):
        oom = 0
        if loss is not None:
            try:
                # backward pass
                loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                else:
                    raise e

        # all-reduce grads and rescale by grad_denom
        if self.args.distributed_world_size > 1:
            grads = [p.grad.data for p in self.model.parameters() if p.requires_grad]
            distributed_utils.all_reduce_and_rescale_tensors(grads, grad_denom)
        else:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.grad.data.div_(grad_denom)

        # clip grads
        if self.args.clip_norm > 0:
            grad_norm = utils.item(torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_norm))
        else:
            grad_norm = math.sqrt(sum(p.grad.data.norm()**2 for p in self.model.parameters()))

        # take an optimization step
        self.optimizer.step()
        self._num_updates += 1

        # update learning rate
        self.lr_scheduler.step_update(self._num_updates)

        return grad_norm, oom

    def valid_step(self, sample):
        """Do forward pass in evaluation mode."""

        sample = self._prepare_sample(sample, volatile=True)

        # forward pass
        loss, sample_sizes, logging_outputs, ooms_fwd = self._forward(sample, eval=True)
        assert not ooms_fwd, 'Ran out of memory during validation'

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

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

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

    def _prepare_sample(self, sample, volatile):
        if sample is None or len(sample) == 0:
            return None
        if hasattr(torch.cuda, 'empty_cache'):
            # clear the caching allocator if this is the largest sample we've seen
            if sample['target'].size(0) > self._max_bsz_seen:
                self._max_bsz_seen = sample['target'].size(0)
                torch.cuda.empty_cache()
        return utils.make_variable(sample, volatile=volatile, cuda=True)
