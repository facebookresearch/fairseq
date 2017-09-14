# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import os
import signal
import threading
from torch import multiprocessing


class MultiprocessingEventLoop(object):
    """Start a multiprocessing event loop."""

    def __init__(self, device_ids=None, multiprocessing_method='spawn'):
        super().__init__()
        self.device_ids = tuple(device_ids)
        self.num_replicas = len(device_ids)
        self.rank = None

        self._mp = multiprocessing.get_context(multiprocessing_method)

        self._start_error_handler()
        self._start_multiprocessing()

    def call_async(self, rank, action, **kwargs):
        """Asynchronously call a function in each child process.

        Call a function named `action` on the rank'th process and return
        a Future with the result.
        """

        def result_generator():
            yield self.return_pipes[rank].recv()

        assert not self.return_pipes[rank].poll(), \
            'return pipe must be consumed before calling another function'
        self.input_pipes[rank].send((action, kwargs))

        return Future(result_generator())

    def stop(self, interrupt_children=False):
        """Stop multiprocessing."""
        for rank in range(self.num_replicas):
            self.input_pipes[rank].close()
            self.return_pipes[rank].close()
            if interrupt_children:
                # send KeyboardInterrupt to children
                os.kill(self.procs[rank].pid, signal.SIGINT)
            else:
                self.procs[rank].join()
        self.error_queue.put((None, None))  # poison pill

    def _start_error_handler(self):
        """Error handler to catch exceptions in child processes."""
        # create a thread to listen for errors in the child processes
        self.error_queue = self._mp.SimpleQueue()
        error_thread = threading.Thread(target=self._error_listener,
                                        daemon=True)
        error_thread.start()

        # create signal handler that executes in the main process/thread and
        # handles errors from child processes
        signal.signal(signal.SIGUSR1, self._signal_handler)

    def _error_listener(self):
        """A thread that listens for errors in the child processes.

        Errors are handled in a signal handler in the main thread.
        """
        (rank, original_trace) = self.error_queue.get()
        if rank is None:  # poison pill, return
            return

        # requeue error and switch to main thread for handling the error
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def _signal_handler(self, signal, frame):
        """Signal handler that handles errors from child processes.

        This signal handler executes in the main/process thread.
        """
        self.stop(interrupt_children=True)
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)

    def _start_multiprocessing(self):
        """Create child processes to run async event loop.

        Each process reads input from a Pipe, performs some computation,
        and returns its output to another Pipe.
        """
        # create child processes
        input_pipes = []
        return_pipes = []
        procs = []
        for rank, id in enumerate(self.device_ids):
            recv_input_pipe, send_input_pipe = self._mp.Pipe(duplex=False)
            recv_return_pipe, send_return_pipe = self._mp.Pipe(duplex=False)
            proc = self._mp.Process(
                target=self._process_event_loop,
                args=(rank, id, recv_input_pipe, send_return_pipe),
                daemon=True)
            proc.start()
            input_pipes.append(send_input_pipe)
            return_pipes.append(recv_return_pipe)
            procs.append(proc)
        self.input_pipes = input_pipes
        self.return_pipes = return_pipes
        self.procs = procs

    def _process_event_loop(self, rank, device_id, input_pipe, return_pipe):
        """Event loop that runs in each child process.

        Event loop:
        - take an action from the input pipe
        - call the corresponding function in this process
        - put the return value in the return pipe

        Any exceptions are put in the error queue.
        """
        self.rank = rank
        try:
            # event loop
            while True:
                action, kwargs = input_pipe.recv()
                action_fn = getattr(self, action)
                return_pipe.send(action_fn(rank, device_id, **kwargs))
        except EOFError:
            # input pipe was closed, do nothing
            pass
        except KeyboardInterrupt:
            # killed by parent, do nothing
            pass
        except Exception:
            # propagate exception from child to parent process, keeping
            # original traceback
            import traceback
            self.error_queue.put((rank, traceback.format_exc()))
        finally:
            # cleanup pipes
            input_pipe.close()
            return_pipe.close()


class Future(object):
    """A wrapper around a Python generator, with syntactic sugar."""
    def __init__(self, generator):
        self.generator = generator

    def gen(self):
        return next(self.generator)

    @staticmethod
    def gen_list(gens):
        return [g.gen() for g in gens]

    @staticmethod
    def gen_tuple_list(gens):
        list = [g.gen() for g in gens]
        return zip(*list)
