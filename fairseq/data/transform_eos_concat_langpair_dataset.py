# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torch.utils.data.dataloader import default_collate

from fairseq.data import ConcatDataset

logger = logging.getLogger(__name__)


class TransformEosConcatLangPairDataset(ConcatDataset):
    """
    It is a combination of TransformEosLangPairDataset and ConcatDataset for multiple LangPairDataset datasets.
    Assume all datasets share the same src_eos, tgt_bos, left_pad_source and left_pad_target
    """

    def __init__(
        self,
        datasets,
        src_eos,
        tgt_bos,
        new_src_eos=None,
        new_tgt_bos=None,
    ):
        super().__init__(datasets)
        if new_src_eos is not None and new_src_eos != []:
            assert len(new_src_eos) == len(datasets)
        else:
            new_src_eos = []
        if new_tgt_bos is not None and new_tgt_bos != []:
            assert len(new_tgt_bos) == len(datasets)
        else:
            new_tgt_bos = []
        self.src_eos = src_eos
        self.tgt_bos = tgt_bos
        self.new_src_eos = (
            torch.LongTensor(new_src_eos).cpu() if len(new_src_eos) > 0 else []
        )
        self.new_tgt_bos = (
            torch.LongTensor(new_tgt_bos).cpu() if len(new_tgt_bos) > 0 else []
        )
        self.left_pad_source = self.is_left_pad_source(datasets)
        self.left_pad_target = self.is_left_pad_target(datasets)
        self.pad_idx = self.src_dict_pad()

    def src_dict_pad(self):
        if hasattr(self.datasets[0], "src_dict"):
            return self.datasets[0].src_dict.pad()
        if hasattr(self.datasets[0], "dataset"):
            return self.datasets[0].dataset.src_dict.pad()
        raise NotImplementedError("No src_dict is found")

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return dataset_idx, self.datasets[dataset_idx][sample_idx]

    def is_left_pad_source(self, datasets):
        def _left_pad_source(ds):
            if hasattr(ds, "left_pad_source"):
                return ds.left_pad_source
            if hasattr(ds, "dataset"):
                return _left_pad_source(ds.dataset)
            logger.warn(f"{type(ds)} has no left_pad_source, using default True")
            return True

        left_pad_source = _left_pad_source(datasets[0])
        for ds in datasets:
            if left_pad_source != _left_pad_source(ds):
                raise ValueError("Different left_pad_source setting detected!")
        return left_pad_source

    def is_left_pad_target(self, datasets):
        def _left_pad_target(ds):
            if hasattr(ds, "left_pad_target"):
                return ds.left_pad_target
            if hasattr(ds, "dataset"):
                return _left_pad_target(ds.dataset)
            logger.warn(f"{type(ds)} has no left_pad_target, using default False")
            return False

        left_pad_target = _left_pad_target(datasets[0])
        for ds in datasets:
            if left_pad_target != _left_pad_target(ds):
                raise ValueError("Different left_pad_target setting detected!")
        return left_pad_target

    def collater(self, samples, **extra_args):
        if len(samples) == 0:
            return samples

        dataset_ids = [s[0] for s in samples]
        samples = [s[1] for s in samples]

        if hasattr(self.datasets[0], "collater"):
            samples = self.datasets[0].collater(samples, **extra_args)
        else:
            samples = default_collate(samples, **extra_args)

        if len(self.new_src_eos) > 0:
            if self.left_pad_source:
                assert (
                    samples["net_input"]["src_tokens"][:, -1] != self.src_eos
                ).sum() == 0
                samples["net_input"]["src_tokens"][:, -1] = self.new_src_eos[
                    dataset_ids
                ]

            else:
                eos_idx = samples["net_input"]["src_lengths"] - 1
                assert (
                    samples["net_input"]["src_tokens"][
                        torch.arange(eos_idx.size(0)), eos_idx
                    ]
                    != self.src_eos
                ).sum() == 0
                samples["net_input"]["src_tokens"].scatter_(
                    1, eos_idx.view(-1, 1), self.new_src_eos[dataset_ids].view(-1, 1)
                )

        if len(self.new_tgt_bos) > 0 and "prev_output_tokens" in samples["net_input"]:
            if self.left_pad_target:
                # TODO: support different padding direction on target side
                raise NotImplementedError(
                    "TransformEosLangPairDataset does not implement --left-pad-target True option"
                )
            else:
                assert (
                    samples["net_input"]["prev_output_tokens"][:, 0] != self.tgt_bos
                ).sum() == 0
                samples["net_input"]["prev_output_tokens"][:, 0] = self.new_tgt_bos[
                    dataset_ids
                ]

        return samples
