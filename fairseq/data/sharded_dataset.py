# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import random

from . import BaseWrapperDataset
from fairseq.data import data_utils


class ShardedDataset(BaseWrapperDataset):
    """A :class:`~fairseq.data.FairseqDataset` wrapper that appends/prepends/strips EOS.

    Loads a dataset which has been sharded into multiple files. each shard is only loaded for each specific epoch

    """

    def __init__(
        self,
        dictionary,
        dataset_impl: str,
        path: str,
        split: str,
        epoch: int,
        name: str = None,
        combine: bool = False,
        seed: int = 0,
    ):
        self._name = name if name is not None else os.path.basename(path)
        num_shards = 0
        for i in itertools.count():
            if not os.path.exists(os.path.join(path, "shard" + str(i))):
                break
            num_shards += 1

        if num_shards > 0 and split == "train":
            random.seed(seed ^ epoch)
            shard = random.randint(0, num_shards - 1)
            split_path = os.path.join(path, "shard" + str(shard), split)
        else:
            split_path = os.path.join(path, split)
            if os.path.isdir(split_path):
                split_path = os.path.join(split_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, dictionary, dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        super().__init__(dataset)

    @property
    def name(self):
        return self._name
