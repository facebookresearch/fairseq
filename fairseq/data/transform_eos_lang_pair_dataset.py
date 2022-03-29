# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch

from . import FairseqDataset


class TransformEosLangPairDataset(FairseqDataset):
    """A :class:`~fairseq.data.FairseqDataset` wrapper that transform bos on
    collated samples of language pair dataset.

    Note that the transformation is applied in :func:`collater`.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset that collates sample into
            LanguagePairDataset schema
        src_eos (int): original source end-of-sentence symbol index to be replaced
        new_src_eos (int, optional): new end-of-sentence symbol index to replace source eos symbol
        tgt_bos (int, optional): original target beginning-of-sentence symbol index to be replaced
        new_tgt_bos (int, optional): new beginning-of-sentence symbol index to replace at the
            beginning of 'prev_output_tokens'
    """

    def __init__(
        self,
        dataset: FairseqDataset,
        src_eos: int,
        new_src_eos: Optional[int] = None,
        tgt_bos: Optional[int] = None,
        new_tgt_bos: Optional[int] = None,
    ):
        self.dataset = dataset
        self.src_eos = src_eos
        self.new_src_eos = new_src_eos
        self.tgt_bos = tgt_bos
        self.new_tgt_bos = new_tgt_bos

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples, **extra_args):
        samples = self.dataset.collater(samples, **extra_args)
        if len(samples) == 0:
            return samples

        if "net_input" not in samples:
            return samples

        if self.new_src_eos is not None:
            if self.dataset.left_pad_source:
                assert (
                    samples["net_input"]["src_tokens"][:, -1] != self.src_eos
                ).sum() == 0
                samples["net_input"]["src_tokens"][:, -1] = self.new_src_eos
            else:
                eos_idx = samples["net_input"]["src_lengths"] - 1
                assert (
                    samples["net_input"]["src_tokens"][
                        torch.arange(eos_idx.size(0)), eos_idx
                    ]
                    != self.src_eos
                ).sum() == 0
                eos_idx = eos_idx.resize_(len(samples["net_input"]["src_lengths"]), 1)
                samples["net_input"]["src_tokens"].scatter_(
                    1, eos_idx, self.new_src_eos
                )

        if (
            self.new_tgt_bos is not None
            and "prev_output_tokens" in samples["net_input"]
        ):
            if self.dataset.left_pad_target:
                # TODO: support different padding direction on target side
                raise NotImplementedError(
                    "TransformEosLangPairDataset does not implement --left-pad-target True option"
                )
            else:
                assert (
                    samples["net_input"]["prev_output_tokens"][:, 0] != self.tgt_bos
                ).sum() == 0
                samples["net_input"]["prev_output_tokens"][:, 0] = self.new_tgt_bos

        return samples

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    @property
    def sizes(self):
        # dataset.sizes can be a dynamically computed sizes:
        return self.dataset.sizes

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
