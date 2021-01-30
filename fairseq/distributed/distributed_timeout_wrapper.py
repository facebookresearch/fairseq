# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import signal
import threading

from torch import nn


logger = logging.getLogger(__name__)


class DistributedTimeoutWrapper(nn.Module):
    """
    A wrapper that kills the process if no progress is made within a given
    *timeout*. The timer is reset every time :func:`forward` is called.

    Usage::

        module = DistributedTimeoutWrapper(module, timeout=30)
        x = module(input)
        time.sleep(20)  # safe
        x = module(input)
        time.sleep(45)  # job will be killed before this returns

    Args:
        module (nn.Module): module to wrap
        timeout (int): number of seconds before killing the process
            (set to a value <= 0 to disable the timeout)
        signal (Optional): signal to send once timeout is triggered
    """
    def __init__(self, module: nn.Module, timeout: int, signal=signal.SIGKILL):
        super().__init__()
        self.module = module
        self.timeout = timeout
        self.signal = signal

        if timeout > 0:
            self._heartbeat = threading.Event()
            self._heartbeat_thread = threading.Thread(
                target=self._check_heartbeat,
                args=(os.getpid(),),
                daemon=True,
            )
            self._heartbeat_thread.start()
            self._terminated = False
        else:
            self._heartbeat = None
            self._heartbeat_thread = None

    def __del__(self):
        self.stop_timeout()

    def __getattr__(self, name):
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def stop_timeout(self):
        if self._heartbeat_thread is not None:
            self._terminated = True
            self._heartbeat_thread.join()

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self._heartbeat is not None:
            self._heartbeat.set()
        return self.module(*args, **kwargs)

    def _check_heartbeat(self, parent_pid):
        self._heartbeat.wait()  # wait for the first forward pass
        while True:
            self._heartbeat.clear()
            success = self._heartbeat.wait(timeout=self.timeout)
            if self._terminated:
                break
            elif not success:
                logger.error((
                    "Killing job for not making progress in {} seconds. "
                    "Set --heartbeat-timeout=-1 to disable this timeout."
                ).format(int(self.timeout)))
                os.kill(parent_pid, self.signal)
                return
