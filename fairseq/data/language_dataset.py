# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools

import numpy as np
import torch

from fairseq.data.data_utils import numpy_seed, uneven_batches_by_size, mask_batches, batches_by_size


class LanguageDatasets(object):
    def __init__(self, src, dst, src_dict, dst_dict):
        self.src = src
        self.dst = dst
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.splits = {}

        assert self.src_dict.pad() == self.dst_dict.pad()
        assert self.src_dict.eos() == self.dst_dict.eos()
        assert self.src_dict.unk() == self.dst_dict.unk()

    def train_dataloader_generator(
            self, split, max_tokens=None, max_sentences=None,
            max_positions=(1024, 1024), seed=None, sample_without_replacement=0,
            shard_id=0, num_shards=1
    ):
        dataset = self.splits[split]
        with numpy_seed(seed):
            batches = uneven_batches_by_size(
                dataset.src, dataset.dst, max_tokens=max_tokens,
                max_sentences=max_sentences, max_positions=max_positions,
                # FP16: during training keep the batch size a multiple of 8
                required_batch_size_multiple=8,
            )
            frozen_batches = tuple(batches)  # freeze

        def dataloader(b):
            b = mask_batches(b, shard_id=shard_id, num_shards=num_shards)  # shard dataset
            return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=b)

        for epoch in itertools.count(1):
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with numpy_seed(seed + epoch):
                batches = list(frozen_batches)  # copy
                np.random.shuffle(batches)
                if sample_without_replacement > 0:
                    # emit sub-epoch dataloaders
                    while len(batches) >= sample_without_replacement:
                        sampled_batches = batches[:sample_without_replacement]
                        remaining_batches = batches[sample_without_replacement:]
                        yield dataloader(sampled_batches)
                        batches = remaining_batches
                    if len(batches) > 0:
                        yield dataloader(batches)
                else:
                    # emit full dataloader
                    yield dataloader(batches)

    def eval_dataloader(self, split, num_workers=0, max_tokens=None,
                        max_sentences=None, max_positions=(1024, 1024),
                        skip_invalid_size_inputs_valid_test=False,
                        descending=False, shard_id=0, num_shards=1):
        dataset = self.splits[split]
        batch_sampler = batches_by_size(
            dataset.src, dataset.dst, max_tokens, max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs_valid_test,
            descending=descending,
            allow_different_src_lens=True)
        batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)
        return torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, collate_fn=dataset.collater,
            batch_sampler=batch_sampler)