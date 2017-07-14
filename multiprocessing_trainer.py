'''
Train a network on multiple GPUs using multiprocessing.
'''
import math
import torch
from torch import multiprocessing, nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import nccl
import utils
from nag import NAG


class MultiprocessingTrainer(object):
    '''Main class for multi-GPU training.

    Each GPU has a full copy of the model and is assigned to its own Python
    Process. Gradients are accumulated with all-reduce and all model replicas
    are updated synchronously after each batch.

    The methods in this class are divided into synchronous functions, which
    prepare and dispatch the input to each Process, and asynchronous functions
    (prefixed with `_async_`), which run on each Process in parallel.
    '''

    def __init__(self, args, model, device_ids=None,
                 multiprocessing_method='forkserver'):
        super(MultiprocessingTrainer, self).__init__()
        self.args = args

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        if device_ids is None:
            device_ids = tuple(range(torch.cuda.device_count()))
        self.device_ids = tuple(device_ids)
        self.num_replicas = len(device_ids)
        self.nccl_uid = nccl.get_unique_id()

        self._start_multiprocessing(multiprocessing_method)
        self._replicate_model(model)

    def _start_multiprocessing(self, method):
        '''Create a Process for each GPU that reads input from a Pipe, performs
        some computation, and returns its output to another Pipe.'''
        self.mp = multiprocessing.get_context(method)
        self.input_pipes = []
        self.return_pipes = []
        self.procs = []
        for rank, id in enumerate(self.device_ids):
            recv_input_pipe, send_input_pipe = self.mp.Pipe(duplex=False)
            recv_return_pipe, send_return_pipe = self.mp.Pipe(duplex=False)
            proc = self.mp.Process(
                target=self._main_process_loop,
                args=(rank, id, recv_input_pipe, send_return_pipe))
            proc.start()
            self.input_pipes.append(send_input_pipe)
            self.return_pipes.append(recv_return_pipe)
            self.procs.append(proc)

    def _replicate_model(self, model):
        '''Create a copy of the model on each GPU/Process.'''
        if not all(p.is_cuda for p in model.parameters()):
            model = model.cuda(self.device_ids[0])
        self.models = nn.parallel.replicate(model, self.device_ids)
        self.models[0] = model
        for rank in range(self.num_replicas):
            self.input_pipes[rank].send(self.models[rank])

    def _scatter_sample(self, sample, volatile=False):
        '''Split and distribute a sample across GPUs.'''
        # prepare input on CPU and let scatter move it to GPU
        sample = utils.prepare_sample(sample, use_cuda=False, volatile=volatile)

        # scatter net inputs across GPUs
        return nn.parallel.scatter(sample['net_input'], self.device_ids)


    def get_model(self):
        '''Get one of the model replicas.'''
        # just return the first model, since all replicas are the same
        return self.models[0]


    def save_checkpoint(self, save_dir, epoch, batch_offset, val_loss=None):
        '''Save a checkpoint for the current model.'''
        self._call_async(0, '_async_save_checkpoint', save_dir=save_dir,
                         epoch=epoch, batch_offset=batch_offset,
                         val_loss=val_loss).gen()

    def _async_save_checkpoint(self, rank, device_id, save_dir, epoch,
                               batch_offset, val_loss):
        utils.save_checkpoint(save_dir, epoch, batch_offset, self.model,
                              self.optimizer, self.lr_scheduler, val_loss)


    def load_checkpoint(self, filename):
        '''Load a checkpoint into the model replicas in each Process.'''
        results = Future.gen_list([
            self._call_async(rank, '_async_load_checkpoint', filename=filename)
            for rank in range(self.num_replicas)
        ])
        epoch, batch_offset = results[0]
        return epoch, batch_offset

    def _async_load_checkpoint(self, rank, device_id, filename):
        return utils.load_checkpoint(filename, self.model, self.optimizer,
                                     self.lr_scheduler)


    def train_step(self, sample):
        '''Scatter a sample and perform forward pass, backward pass and take
        gradient steps in each Process.'''
        # scatter sample across GPUs
        net_inputs = self._scatter_sample(sample)

        # forward pass, backward pass and gradient step
        losses = [
            self._call_async(rank, '_async_train_step', net_input=input,
                             grad_denom=sample['ntokens'])
            for rank, input in enumerate(net_inputs)
        ]

        # accumulate and normalize loss
        loss = sum(Future.gen_list(losses)) / sample['ntokens']

        return loss / math.log(2)

    def _async_train_step(self, rank, device_id, net_input, grad_denom):
        self.model.train()

        # zero grads even if net_input is None, since we will all-reduce them
        self.optimizer.zero_grad()

        # calculate loss and grads
        loss = 0
        if net_input is not None:
            loss_ = self.model(**net_input)
            loss_.backward()
            loss = loss_.data[0]

        # flatten grads into a contiguous block of memory
        if self.flat_grads is None:
            self.flat_grads = self._flatten_grads_(self.model)

        # all-reduce grads
        nccl.all_reduce(self.flat_grads)

        # normalize and clip grads
        self.flat_grads.div_(grad_denom)
        self._clip_grads_(self.flat_grads, self.args.clip_norm)

        # take an optimization step
        self.optimizer.step()

        return loss

    def _flatten_grads_(self, model):
        num_params = sum(p.data.numel() for p in model.parameters())
        flat_grads = next(model.parameters()).data.new(num_params)
        offset = 0
        for p in model.parameters():
            grad = p.grad.data
            numel, sz = grad.numel(), grad.size()
            flat_grads[offset:offset+numel] = grad.view(-1)
            grad.set_(flat_grads[offset:offset+numel])
            grad.resize_(sz)  # preserve original shape
            offset += numel
        return flat_grads

    def _clip_grads_(self, flat_grads, clipv):
        if clipv > 0:
            norm = flat_grads.norm()
            if norm > clipv:
                coef = max(norm, 1e-6) / clipv
                flat_grads.div_(coef)
        return flat_grads


    def valid_step(self, sample):
        '''Scatter a sample and perform forward in parallel.'''
        # forward pass
        net_inputs = self._scatter_sample(sample, volatile=True)
        losses = [
            self._call_async(rank, '_async_valid_step', net_input=input)
            for rank, input in enumerate(net_inputs)
        ]

        # accumulate and normalize loss
        loss = sum(Future.gen_list(losses)) / sample['ntokens']

        return loss / math.log(2)

    def _async_valid_step(self, rank, device_id, net_input):
        if net_input is None:
            return 0
        self.model.eval()
        loss = self.model(**net_input)
        return loss.data[0]


    def get_lr(self):
        '''Get the current learning rate.'''
        return self._call_async(0, '_async_get_lr').gen()

    def _async_get_lr(self, rank, device_id):
        return self.optimizer.param_groups[0]['lr']


    def lr_step(self, val_loss=None, epoch=None):
        '''Adjust the learning rate depending on the validation loss.'''
        lr = Future.gen_list([
            self._call_async(rank, '_async_lr_step', val_loss=val_loss,
                             epoch=epoch)
            for rank in range(self.num_replicas)
        ])
        return lr[0]

    def _async_lr_step(self, rank, device_id, epoch, val_loss):
        # update the learning rate
        if self.args.force_anneal > 0:
            self.lr_scheduler.step(epoch)
        else:
            self.lr_scheduler.step(val_loss, epoch)
        return self.optimizer.param_groups[0]['lr']


    def _call_async(self, rank, action, **kwargs):
        '''Call a function named `action` on the rank'th process and return a
        Future with the result.'''
        assert not self.return_pipes[rank].poll(), \
            'return pipe must be consumed before calling another function'
        self.input_pipes[rank].send((action, kwargs))
        def result_generator():
            yield self.return_pipes[rank].recv()
        return Future(result_generator())


    def _main_process_loop(self, rank, device_id, input_pipe, return_pipe):
        '''Main loop run in each Process that reads inputs from an input Pipe,
        performs some computation, and returns the output to another Pipe.'''
        with torch.cuda.device(device_id):
            # initialize NCCL
            nccl.initialize(self.num_replicas, self.nccl_uid, device_id)

            # get replicated model
            self.model = input_pipe.recv()

            # initialize optimizer
            self.optimizer = NAG(self.model.parameters(), lr=self.args.lr,
                                 momentum=self.args.momentum,
                                 weight_decay=self.args.weight_decay)
            self.flat_grads = None

            # initialize LR scheduler
            self.lr_scheduler = self._build_lr_scheduler()

            # main action loop:
            # - take actions from the input Pipe
            # - call the corresponding function on this Process
            # - put the return value in the return Pipe
            while True:
                action, kwargs = input_pipe.recv()
                action_fn = getattr(self, action)
                return_pipe.send(action_fn(rank, device_id, **kwargs))

    def _build_lr_scheduler(self):
        if self.args.force_anneal > 0:
            def anneal(e):
                if e < self.args.force_anneal:
                    return 1
                else:
                    return 0.1 ** (e + 1 - self.args.force_anneal)
            lr_scheduler = LambdaLR(self.optimizer, anneal)
            lr_scheduler.best = None
        else:
            # decay the LR by 0.1 every time the validation loss plateaus
            lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=0)
        return lr_scheduler


class Future(object):
    '''A simple wrapper around a Python generator that provides cleaner
    generation syntax.'''
    def __init__(self, generator):
        self.generator = generator

    def gen(self):
        return next(self.generator)

    @staticmethod
    def gen_list(gens):
        return [g.gen() for g in gens]
