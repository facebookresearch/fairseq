# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import io
import logging
import os

import torch

from .raw_audio_dataset import RawAudioDataset


logger = logging.getLogger(__name__)


class EverstoreAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
        )

        self.gets = 0
        self.fails = 0

        self.handles = []
        self.starts = None
        with open(manifest_path, "r") as f:
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2 or len(items) == 3, line
                if len(items) == 2:
                    length = int(float(items[1]))
                    if length > 0:
                        self.handles.append(items[0])
                        self.sizes.append(length)
                else:
                    if self.starts is None:
                        self.starts = []
                    s = int(items[1])
                    e = int(items[2])
                    length = e - s
                    if length > 0:
                        self.handles.append(items[0])
                        self.sizes.append(length)
                        self.starts.append(s)

        assert (self.starts is None) or (len(self.starts) == len(self.handles))

        import common.thread_safe_fork

        common.thread_safe_fork.threadSafeForkRegisterAtFork()

        self.clients = {}  # this will leak every epoch!!! is cleared in prefetch

    def __getitem__(self, index):
        from .fb_soundfile import read as sf_read

        pid = os.getpid()
        if pid not in self.clients:
            from langtech import EverstoreIo

            self.clients[pid] = EverstoreIo.EverstoreIo()
        success, blob = self.clients[pid].get(self.handles[index], "wav2vec")
        self.gets += 1
        if not success:
            self.fails += 1
            if self.fails % 10000 == 0:
                logger.warning(
                    f"failed {self.fails} times out of {self.gets} ({100*self.fails/self.gets}%)"
                )
            return {"id": index, "source": None}

        wav, curr_sample_rate = sf_read(io.BytesIO(blob))

        if self.starts is not None:
            s = self.starts[index]
            e = s + self.sizes[index]
            wav = wav[s:e]

        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {"id": index, "source": feats}
