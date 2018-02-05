# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

"""
Train a network on multiple GPUs using multiprocessing.
"""

import math
import torch
import torch.distributed

from fairseq import optim
from fairseq.optim import lr_scheduler
from fairseq import utils
from fairseq.tcp_connector import TcpConnector


class Trainer(object):
    """Main class for multi-GPU training.

    Each GPU has a full copy of the model and is assigned to its own Python
    process. Gradients are accumulated with all-reduce and all model replicas
    are updated synchronously after each batch.

    The methods in this class are divided into synchronous functions, which
    prepare and dispatch the input to each process, and asynchronous functions
    (prefixed with `_async_`), which run on each process in parallel.
    """

    def __init__(self, args, model, criterion):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')


        self.criterion = criterion
        self.args = args
        self.rank = args.distributed_rank
        self.world_size = args.distributed_world_size

        self._init_tcp_connector(args)

        # copy model and criterion to current device
        self.model = model.cuda()
        self.criterion = criterion.cuda()

        # initialize optimizer and LR scheduler
        self.optimizer = optim.build_optimizer(self.args, self.model.parameters())
        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

        self.loss = None
        self._max_bsz_seen = 0
        self._num_updates = 0

    def _init_tcp_connector(self, args):
        """Discover rank of current host and hostnames of all other hosts."""
        if args.distributed_world_size > 1:
            self.tcp_connector = TcpConnector(
                args.distributed_port, self.rank, self.world_size, args.distributed_master_host)

    def get_model(self):
        """Get one of the model replicas."""
        # just return the first model, since all replicas are the same
        return self.model

    def save_checkpoint(self, filename, extra_state):
        if self.rank == 0:  # only save one checkpoint
            utils.save_state(filename, self.args, self.model, self.criterion, self.optimizer,
                             self.lr_scheduler, self._num_updates, self._optim_history, extra_state)

    def load_checkpoint(self, filename):
        """Load a checkpoint into the model replicas in each process."""
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

    def set_seed(self, seed):
        torch.manual_seed(seed)

    def train_step(self, sample):
        """Do forward, backward and gradient step in parallel."""

        self.prepare_sample(sample, volatile=False)

        # forward pass
        sample_sizes, logging_outputs, ooms_fwd = self.forward()

        if self.world_size > 1:
            # synchronize logging outputs for multi-node training
            sample_sizes, logging_outputs = zip(*list(self.tcp_connector.all_gather((sample_sizes, logging_outputs))))
        else:
            sample_sizes = [sample_sizes]
            logging_outputs = [logging_outputs]

        # backward pass, all-reduce gradients and take an optimization step
        grad_denom = self.criterion.__class__.grad_denom(sample_sizes)
        grad_norm, ooms_bwd, lr = self.backward_and_opt(grad_denom=grad_denom)

        # aggregate logging output
        logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)
        logging_output['lr'] = lr
        logging_output['gnorm'] = grad_norm  # log the gradient norm
        logging_output['oom'] = ooms_fwd + ooms_bwd
        logging_output['ntokens'] = sum(log.get('ntokens', 0) for log in logging_outputs)
        logging_output['nsentences'] = sum(log.get('nsentences', 0) for log in logging_outputs)

        return logging_output

    def forward(self, eval=False):
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        with utils.maybe_no_grad(eval):
            self.sample_size = 0
            logging_output = {'ntokens': 0, 'nsentences': 0}
            oom = False
            if self._sample is not None:
                try:
                    # calculate loss and sample size
                    self.loss, self.sample_size, logging_output = self.criterion(self.model, self._sample)
                    logging_output['ntokens'] = self._sample['ntokens']
                    logging_output['nsentences'] = self._sample['target'].size(0)
                except RuntimeError as e:
                    if not eval and 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, skipping batch')
                        oom = True
                        self.loss = None
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e

        return self.sample_size, logging_output, oom

    def backward_and_opt(self, grad_denom):
        oom = False
        if self.loss is not None:
            try:
                # backward pass
                self.loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = True
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                else:
                    raise e

        if self.world_size > 1:
            # all-reduce grads and rescale by grad_denom
            self._all_reduce_and_rescale_grads(grad_denom)
        else:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.grad.data.div_(grad_denom)

        # clip grads
        if self.args.clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_norm)
        else:
            grad_norm = math.sqrt(sum([p.grad.data.norm()**2 for p in self.model.parameters()]))

        # take an optimization step
        self.optimizer.step()
        self._num_updates += 1

        # update learning rate
        lr = self.lr_scheduler.step_update(self._num_updates)

        # reset loss
        self.loss = None

        return grad_norm, oom, lr

    def _all_reduce_and_rescale_grads(self, grad_denom, buffer_size=10485760):
        """All-reduce and rescale gradients in chunks of the specified size."""
        grads = [p.grad.data for p in self.model.parameters() if p.requires_grad]
        buffer_t = grads[0].new(math.ceil(buffer_size / grads[0].element_size())).zero_()
        buffer = []

        def all_reduce_buffer():
            # copy grads into buffer_t
            offset = 0
            for g in buffer:
                numel = g.numel()
                buffer_t[offset:offset+numel].copy_(g.view(-1))
                offset += numel
            # all-reduce and rescale
            torch.distributed.all_reduce(buffer_t[:offset])
            buffer_t.div_(grad_denom)

            # copy all-reduced buffer back into grads
            offset = 0
            for g in buffer:
                numel = g.numel()
                g.view(-1).copy_(buffer_t[offset:offset+numel])
                offset += numel

        filled = 0
        for g in grads:
            sz = g.numel() * g.element_size()
            if sz > buffer_size:
                # grad is bigger than buffer, all-reduce and rescale directly
                torch.distributed.all_reduce(g)
                g.div_(grad_denom)
            elif filled + sz > buffer_size:
                # buffer is full, all-reduce and replace buffer with grad
                all_reduce_buffer()
                buffer = [g]
                filled = sz
            else:
                # add grad to buffer
                buffer.append(g)
                filled += sz
        if len(buffer) > 0:
            all_reduce_buffer()

    def valid_step(self, sample):
        """Do forward pass in parallel."""
        # scatter sample across GPUs
        self.prepare_sample(sample, volatile=True)

        # forward pass

        _sample_sizes, logging_outputs, ooms_fwd = self.forward(eval=True)
        assert not ooms_fwd

        if self.world_size > 1:
            logging_outputs = list(self.tcp_connector.all_gather(logging_outputs))
        else:
            logging_outputs = [logging_outputs]

        # aggregate logging output
        logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)
        logging_output['ntokens'] = sum(log.get('ntokens', 0) for log in logging_outputs)
        logging_output['nsentences'] = sum(log.get('nsentences', 0) for log in logging_outputs)

        return logging_output

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates


    def prepare_sample(self, sample, volatile):
        if sample is None or len(sample) == 0:
            self._sample = None
        else:
            if hasattr(torch.cuda, 'empty_cache'):
                # clear the caching allocator if this is the largest sample we've seen
                if sample['target'].size(0) > self._max_bsz_seen:
                    self._max_bsz_seen = sample['target'].size(0)
                    torch.cuda.empty_cache()

            self._sample = utils.make_variable(sample, volatile=volatile, cuda=True)

    def shutdown(self):
        self.tcp_connector.shutdown()

