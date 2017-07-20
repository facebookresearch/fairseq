'''
Train a network on multiple GPUs using multiprocessing.
'''
import math
import os
import signal
import threading
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
                 multiprocessing_method='spawn'):
        super(MultiprocessingTrainer, self).__init__()
        self.args = args

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        if device_ids is None:
            device_ids = tuple(range(torch.cuda.device_count()))
        self.device_ids = tuple(device_ids)
        self.num_replicas = len(device_ids)
        self.nccl_uid = nccl.get_unique_id()

        self._start_multiprocessing(multiprocessing_method, model)

    def _start_multiprocessing(self, method, model):
        '''
        Create a Process for each GPU that reads input from a Pipe, performs
        some computation, and returns its output to another Pipe. We also start
        an error handling thread to catch exceptions in the children.
        '''
        model = model.share_memory()
        mp = multiprocessing.get_context(method)

        # create a thread to listen for errors in the child processes
        self.error_queue = mp.SimpleQueue()
        error_thread = threading.Thread(target=self._error_listener)
        error_thread.daemon = True
        error_thread.start()

        # create child processes
        input_pipes = []
        return_pipes = []
        procs = []
        for rank, id in enumerate(self.device_ids):
            recv_input_pipe, send_input_pipe = mp.Pipe(duplex=False)
            recv_return_pipe, send_return_pipe = mp.Pipe(duplex=False)
            proc = mp.Process(
                target=self._main_process_loop_safe,
                args=(rank, id, model, recv_input_pipe, send_return_pipe))
            proc.daemon = True
            proc.start()
            input_pipes.append(send_input_pipe)
            return_pipes.append(recv_return_pipe)
            procs.append(proc)
        self.input_pipes = input_pipes
        self.return_pipes = return_pipes
        self.procs = procs

        # create signal handler that executes in the main process/thread and
        # handles errors from child processes
        signal.signal(signal.SIGUSR1, self._signal_handler)


    def _scatter_sample(self, sample, volatile=False):
        '''Split and distribute a sample across GPUs.'''
        # prepare input on CPU and let scatter move it to GPU
        # returned list may be smaller than the number of available devices
        res = [utils.prepare_sample(sample[i],
                                    volatile=volatile,
                                    cuda_device=device_id)
               for i, device_id in zip(range(0, len(sample)), self.device_ids)]
        return res + [None]*(self.num_replicas - len(sample))

    def stop(self, interrupt_children=False):
        '''Stop multiprocessing.'''
        for rank in range(self.num_replicas):
            self.input_pipes[rank].close()
            self.return_pipes[rank].close()
            if interrupt_children:
                # send KeyboardInterrupt to children
                os.kill(self.procs[rank].pid, signal.SIGINT)
            else:
                self.procs[rank].join()
        self.error_queue.put((None, None))  # poison pill


    def get_model(self):
        '''Get one of the model replicas.'''
        # just return the first model, since all replicas are the same
        return self._call_async(0, '_async_get_model').gen()

    def _async_get_model(self, rank, device_id):
        return self.model


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
        ntokens = sum(s['ntokens'] if s else 0 for s in sample)

        # forward pass, backward pass and gradient step
        losses = [
            self._call_async(rank, '_async_train_step',
                             net_input=input['net_input'] if input else None,
                             grad_denom=ntokens)
            for rank, input in enumerate(net_inputs)
        ]

        # accumulate and normalize loss
        loss = sum(Future.gen_list(losses)) / ntokens

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
            self._call_async(rank, '_async_valid_step',
                             net_input=input['net_input'] if input else None)
            for rank, input in enumerate(net_inputs)
        ]

        # accumulate and normalize loss
        loss = sum(Future.gen_list(losses)) / sum(s['ntokens'] if s else 0 for s in sample)

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


    def _main_process_loop_safe(self, rank, device_id, model, input_pipe, return_pipe):
        '''Wraps main process loop in each child Process and sends back exception to the
        parent in case of failures (typically OOMs)'''
        try:
            self._main_process_loop(rank, device_id, model, input_pipe, return_pipe)
        except EOFError:
            # input pipe was closed, do nothing
            pass
        except KeyboardInterrupt:
            # killed by parent, do nothing
            pass
        except Exception as e:
            # propagate exception from child to parent process, keeping original
            # traceback
            import traceback
            self.error_queue.put((rank, traceback.format_exc()))
        finally:
            # cleanup pipes
            input_pipe.close()
            return_pipe.close()

    def _main_process_loop(self, rank, device_id, model, input_pipe, return_pipe):
        '''Main loop run in each Process that reads inputs from an input Pipe,
        performs some computation, and returns the output to another Pipe.'''
        with torch.cuda.device(device_id):
            # initialize NCCL
            nccl.initialize(self.num_replicas, self.nccl_uid, device_id)

            # copy model to current device
            self.model = model.cuda()

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

    def _error_listener(self):
        '''A thread that listens for errors in the child processes. Errors are
        handled in a separate signal handler in the main thread.'''
        (rank, original_trace) = self.error_queue.get()
        if rank is None:  # poison pill, return
            return

        # requeue error and switch to main thread for handling the error
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def _signal_handler(self, signal, frame):
        '''Signal handler that executes in the main process/thread and handles
        errors from child processes.'''
        self.stop(interrupt_children=True)
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)


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
