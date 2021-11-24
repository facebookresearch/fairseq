# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from fairseq import utils
from fairseq.data import (
    Dictionary,
    TokenBlockDataset,
    data_utils,
    iterators,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.distributed import utils as dist_utils
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II


logger = logging.getLogger(__name__)


@dataclass
class TruncatedBPTTLMConfig(FairseqDataclass):
    data: str = field(default="???", metadata={"help": "path to data directory"})
    tokens_per_sample: int = field(
        default=1024, metadata={"help": "max number of tokens per sequence"},
    )
    batch_size: int = II("dataset.batch_size")
    # Some models use *max_target_positions* to know how many positional
    # embeddings to learn. We use II(...) to make it default to
    # *tokens_per_sample*, but in principle there could be more positional
    # embeddings than tokens in a single batch. This may also be irrelevant for
    # custom model implementations.
    max_target_positions: int = II("task.tokens_per_sample")
    # these will be populated automatically if not provided
    data_parallel_rank: Optional[int] = None
    data_parallel_size: Optional[int] = None


@register_task("truncated_bptt_lm", dataclass=TruncatedBPTTLMConfig)
class TruncatedBPTTLMTask(FairseqTask):
    def __init__(self, cfg: TruncatedBPTTLMConfig):
        super().__init__(cfg)

        if cfg.data_parallel_rank is None or cfg.data_parallel_size is None:
            if torch.distributed.is_initialized():
                cfg.data_parallel_rank = dist_utils.get_data_parallel_rank()
                cfg.data_parallel_size = dist_utils.get_data_parallel_world_size()
            else:
                cfg.data_parallel_rank = 0
                cfg.data_parallel_size = 1

        # load the dictionary
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        self.dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        logger.info("dictionary: {} types".format(len(self.dictionary)))

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)"""

        # support sharded datasets
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        # each element of *data* will be a tensorized line from the original
        # text dataset, similar to ``open(split_path).readlines()``
        data = data_utils.load_indexed_dataset(
            split_path, self.dictionary, combine=combine
        )
        if data is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        # this is similar to ``data.view(-1).split(tokens_per_sample)``
        data = TokenBlockDataset(
            data,
            data.sizes,
            block_size=self.cfg.tokens_per_sample,
            pad=None,  # unused
            eos=None,  # unused
            break_mode="none",
        )

        self.datasets[split] = TruncatedBPTTDataset(
            data=data,
            bsz_per_shard=self.cfg.batch_size,
            shard_id=self.cfg.data_parallel_rank,
            num_shards=self.cfg.data_parallel_size,
        )

    def dataset(self, split):
        return self.datasets[split]

    def get_batch_iterator(
        self,
        dataset,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        skip_remainder_batch=False,
        **kwargs
    ):
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            # we don't use the batching functionality from EpochBatchIterator;
            # instead every item in *dataset* is a whole batch
            batch_sampler=[[i] for i in range(len(dataset))],
            disable_shuffling=True,
            skip_remainder_batch=skip_remainder_batch,
        )

    def _collate_fn(self, items: List[List[torch.Tensor]]):
        # we don't use fairseq's batching functionality, so we expect a single
        # Tensor of type List[torch.Tensor]
        assert len(items) == 1

        # item will have shape B x T (the last batch may have length < T)
        id, item = items[0]
        item = data_utils.collate_tokens(item, pad_idx=self.source_dictionary.pad())
        B, T = item.size()

        # shift item one position over and append a padding token for the target
        target = torch.nn.functional.pad(
            item[:, 1:], (0, 1, 0, 0), value=self.target_dictionary.pad()
        )

        # fairseq expects batches to have the following structure
        return {
            "id": torch.tensor([id] * item.size(0)),
            "net_input": {"src_tokens": item,},
            "target": target,
            "nsentences": item.size(0),
            "ntokens": item.numel(),
        }

    def build_dataset_for_inference(
        self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        eos = self.source_dictionary.eos()
        dataset = TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=None,  # ignored for "eos" break mode
            pad=self.source_dictionary.pad(),
            eos=eos,
            break_mode="eos",
        )

        class Dataset(torch.utils.data.Dataset):
            def __getitem__(self, i):
                item = dataset[i]
                if item[-1] == eos:
                    # remove eos to support generating with a prefix
                    item = item[:-1]
                return (i, [item])

            def __len__(self):
                return len(dataset)

        return Dataset()

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            if constraints is not None:
                raise NotImplementedError

            # SequenceGenerator doesn't use *src_tokens* directly, we need to
            # pass the *prefix_tokens* argument instead.
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]

            # begin generation with the end-of-sentence token
            bos_token = self.source_dictionary.eos()

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        context_window: int = 0,
    ):
        if context_window > 0:
            raise NotImplementedError(
                "Transformer-XL doesn't need --context-window, try "
                "--model-overrides '{\"mem_len\":42}' instead "
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        ).next_epoch_itr(shuffle=False)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class TruncatedBPTTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[torch.Tensor],  # ordered list of items
        bsz_per_shard,  # number of items processed per GPUs per forward
        shard_id,  # current GPU ID
        num_shards,  # number of GPUs
    ):
        super().__init__()
        self.data = data

        def batchify(data, bsz):
            # Work out how cleanly we can divide the dataset into bsz parts.
            nbatch = data.size(0) // bsz
            # Trim off any extra elements that wouldn't cleanly fit (remainders).
            data = data.narrow(0, 0, nbatch * bsz)
            # Evenly divide the data across the bsz batches.
            data = data.view(bsz, -1).contiguous()
            return data

        # total number of sequences processed by all GPUs in each forward pass
        global_batch_size = bsz_per_shard * num_shards

        """
        With a 16 item dataset, bsz_per_shard=2 and num_shards=3,
        *indices* might look like:

            indices = [[0, 1],
                       [2, 3],
                       [4, 5],
                       [6, 7],
                       [8, 9],
                       [10, 11]]

        The size of the TruncatedBPTTDataset instance will be 2,
        and shard 1 will see items:

            [(0, [data[4], data[6]]),
             (1, [data[5], data[7]])]
        """
        indices = batchify(torch.arange(len(data)), global_batch_size)
        assert indices.size(0) == global_batch_size

        self.my_indices = indices[
            shard_id * bsz_per_shard : (shard_id + 1) * bsz_per_shard
        ]
        assert self.my_indices.size(0) == bsz_per_shard

    def __len__(self):
        return self.my_indices.size(1)

    def __getitem__(self, i) -> Tuple[int, List[torch.Tensor]]:
        return (i, [self.data[idx] for idx in self.my_indices[:, i]])
