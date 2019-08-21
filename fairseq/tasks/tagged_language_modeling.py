# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import os

from fairseq.data import (
    ConcatDataset,
    data_utils,
    MonolingualDataset,
    PrependDataset,
    ReplaceDataset,
    ShardedDataset,
    SubsampleDataset,
    TokenBlockDataset,
)
from fairseq.tasks import register_task

from fairseq.tasks.language_modeling import LanguageModelingTask


@register_task("tagged_language_modeling")
class TaggedLanguageModelingTask(LanguageModelingTask):
    """
    Like the language modeling task, but prepends tags to each sample
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        LanguageModelingTask.add_args(parser)
        parser.add_argument(
            "--multiple-datasets",
            action="store_true",
            help="if set, treats paths in data as separate datasets to be combined, "
            "rather than as splits of a single dataset",
        )
        parser.add_argument(
            "--prepend-ds-name",
            action="store_true",
            help="if set and multiple-datasets is also set, prepends the name of the ds instead of "
            "bos/eos token",
        )
        parser.add_argument(
            "--generic-ds-name-chance",
            type=float,
            metavar="P",
            default=0,
            help='if multiple datasets is used, sets the prepended ds name to "generic" '
            "this percentage of time",
        )
        parser.add_argument(
            "--subsample-splits",
            type=str,
            metavar="SPLITS",
            default="valid",
            help="if multiple datasets is used, subsamples specified split(colon separated) to "
            "the size of the smallest split",
        )

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)
        self.subsample_splits = (
            set()
            if args.subsample_splits is None
            else set(args.subsample_splits.split(":"))
        )

    def make_prepended_ds(self, dataset):
        def ds_name(dataset, index):
            if (
                self.args.generic_ds_name_chance > 0
                and np.random.rand() <= self.args.generic_ds_name_chance
            ):
                ds_name = "generic"
            else:
                ds_name = dataset.attr("name", index)
            assert ds_name is not None
            return self.dictionary.indices[ds_name]

        dataset = PrependDataset(
            dataset, prepend_getter=ds_name, ensure_first_token_is=self.dictionary.eos()
        )
        return dataset

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(":")
        assert len(paths) > 0

        if self.args.multiple_datasets:
            if len(paths) == 1:
                paths = [os.path.join(paths[0], p) for p in next(os.walk(paths[0]))[1]]
            datasets = [
                ShardedDataset(
                    self.dictionary,
                    self.args.dataset_impl,
                    path,
                    split,
                    epoch,
                    combine=combine,
                )
                for path in paths
            ]

            if split in self.subsample_splits:
                sizes = [sum(d.sizes) for d in datasets]
                min_sz = min(sizes)
                ratios = [min_sz / sz for sz in sizes]
                datasets = [
                    SubsampleDataset(d, r) if r < 1 else d
                    for d, r in zip(datasets, ratios)
                ]

            dataset = ConcatDataset(datasets)
        else:
            data_path = paths[epoch % len(paths)]
            split_path = os.path.join(data_path, split)

            dataset = data_utils.load_indexed_dataset(
                split_path, self.dictionary, self.args.dataset_impl, combine=combine
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
        )

        if self.args.prepend_ds_name:
            dataset = self.make_prepended_ds(dataset)

        dataset = ReplaceDataset(dataset, { self.dictionary.eos(): self.dictionary.indices['\\n'] }, offset=1)

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )

        self.datasets[split] = MonolingualDataset(
            dataset,
            dataset.sizes,
            self.dictionary,
            self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
        )
