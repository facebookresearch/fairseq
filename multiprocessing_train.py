#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import random
import signal
import torch

from fairseq import distributed_utils, options

from train import main as single_process_main


def main(args):
    # Set distributed training parameters for a single node.
    args.distributed_world_size = torch.cuda.device_count()
    args.distributed_init_method = 'tcp://localhost:{port}'.format(
        port=random.randint(10000, 20000))

    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(args.distributed_world_size):
        args.distributed_rank = i
        args.device_id = i
        procs.append(mp.Process(target=run, args=(args, error_queue, ), daemon=True))
        procs[i].start()
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, error_queue):
    try:
        args.distributed_rank = distributed_utils.distributed_init(args)
        single_process_main(args)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.distributed_rank, traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
