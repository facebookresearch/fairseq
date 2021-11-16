# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict, defaultdict
import json
import os
import logging
from argparse import ArgumentError

from fairseq import options, models
from fairseq.data import (
    data_utils,
    Dictionary,
    LanguagePairDataset,
    IndexedDataset,
    FairseqDataset,
)
from .multitask_data_utils import (
    MultitaskDatasetWrapper,
    MultidatasetEpochBatchIterator,
)


from fairseq.tasks import LegacyFairseqTask, register_task

logger = logging.getLogger(__name__)


@register_task("laser")
class LaserTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "configfile", metavar="PATH", help="dataset configuration file in json"
        )
        parser.add_argument(
            "--weighting-alpha",
            type=float,
            default=None,
            help="alpha for automatic weighting",
        )
        parser.add_argument(
            "--raw-text", action="store_true", help="load raw text dataset"
        )
        parser.add_argument(
            "--left-pad-source",
            default="True",
            type=str,
            metavar="BOOL",
            help="pad the source on the left (default: True)",
        )
        parser.add_argument(
            "--left-pad-target",
            default="False",
            type=str,
            metavar="BOOL",
            help="pad the target on the left (default: False)",
        )
        try:
            parser.add_argument(
                "--max-source-positions",
                default=1024,
                type=int,
                metavar="N",
                help="max number of tokens in the source sequence",
            )
            parser.add_argument(
                "--max-target-positions",
                default=1024,
                type=int,
                metavar="N",
                help="max number of tokens in the target sequence",
            )
        except ArgumentError:
            # this might have already been defined. Once we transition this to hydra it should be fine to add it here.
            pass

    def __init__(self, args, config, src_dictionary, tgt_dictionary, num_tasks):
        super().__init__(args)
        self.config = config
        self.src_dictionary = src_dictionary
        self.tgt_dictionary = tgt_dictionary
        self.num_tasks = num_tasks

    @classmethod
    def setup_task(cls, args, **kwargs):
        with open(args.configfile, "r") as f:
            config = json.load(f)
        num_tasks = max(dataset["id"] for dataset in config["train"]) + 1

        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        src_dictionary = Dictionary.load(config["src_vocab"])
        tgt_dictionary = Dictionary.load(config["tgt_vocab"])

        logger.info(
            "| src Dictionary {} : {} types".format(
                config["src_vocab"], len(src_dictionary)
            )
        )
        logger.info(
            "| tgt Dictionary {} : {} types".format(
                config["tgt_vocab"], len(tgt_dictionary)
            )
        )

        return cls(args, config, src_dictionary, tgt_dictionary, num_tasks)

    # Experimental overriding for backtranslation
    def build_model(self, args):
        model = models.build_model(args, self)
        return model

    def dataset(self, split):
        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        return self.datasets[split]

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a dataset split."""

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                raise Exception("Unable to handle raw text.")
            dataset = IndexedDataset(path, fix_lua_indexing=True)

            return dataset

        pair_datasets = OrderedDict()

        if split == "valid":
            self.datasets[split] = pair_datasets
            return

        if split not in self.config:
            raise FileNotFoundError(
                "Dataset not found in config file: {}".format(split)
            )

        size_by_corpus = defaultdict(int)
        size_sum = 0
        size_sum_with_subsampling = 0
        init_pair_datasets = {}

        for dataset_config in self.config[split]:
            src_path = os.path.dirname(dataset_config["src"])
            corpus_name = src_path.split("/")[-2]
            language_pair_name = src_path.split("/")[-1]
            pair_datasets_key = corpus_name + "-" + language_pair_name

            logger.info(f"loading... {pair_datasets_key}")
            if "src" in dataset_config:
                src_dataset = indexed_dataset(
                    dataset_config["src"], self.src_dictionary
                )
            else:
                src_dataset = None

            if "tgt" in dataset_config:
                tgt_dataset = indexed_dataset(
                    dataset_config["tgt"], self.tgt_dictionary
                )
            else:
                tgt_dataset = None

            dataset = LanguagePairDataset(
                src_dataset,
                src_dataset.sizes,
                self.src_dictionary,
                tgt_dataset,
                tgt_dataset.sizes,
                self.tgt_dictionary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
            )

            if pair_datasets_key in init_pair_datasets:
                logger.warning(
                    f"Ignoring already added {pair_datasets_key}. "
                    f"Consider using `sample` key in order to upsample."
                )
            else:
                init_pair_datasets[pair_datasets_key] = {
                    "dataset": dataset,
                    "sample": dataset_config.get("sample", None),
                    "id": dataset_config.get("id", None),
                    "len": len(dataset),
                }

        length_sum = 0
        weighted_freqs_sum = 0
        freq_per_dataset = {}
        vmax = 0
        vmin = 1
        weighted_freq_per_dataset = {}

        if self.args.weighting_alpha:
            for key in init_pair_datasets:
                if init_pair_datasets[key]["sample"] is None:
                    length_sum += len(init_pair_datasets[key]["dataset"])

            for key in init_pair_datasets:
                if init_pair_datasets[key]["sample"] is None:
                    val = float(init_pair_datasets[key]["len"]) / length_sum
                    freq_per_dataset[key] = val
                    weighted_freqs_sum += val ** self.args.weighting_alpha

            for key in freq_per_dataset:
                val = (
                    freq_per_dataset[key] ** self.args.weighting_alpha
                    / weighted_freqs_sum
                )
                vmin = min(vmin, val)
                vmax = max(vmax, val)
                weighted_freq_per_dataset[key] = val

        for pair_datasets_key in init_pair_datasets:
            dataset_config = init_pair_datasets[pair_datasets_key]
            dataset = dataset_config["dataset"]
            sample = dataset_config["sample"]
            if sample is None:
                sample = 1.0

            if pair_datasets_key in weighted_freq_per_dataset:
                w = vmax / weighted_freq_per_dataset[pair_datasets_key]
                sample = w

            sample = round(sample)

            initial_sample = sample
            initial_pair_datasets_key = pair_datasets_key

            while sample >= 1.0:
                assert (
                    pair_datasets_key not in pair_datasets
                ), f"{pair_datasets_key} already in"
                size_sum_with_subsampling += len(dataset)
                pair_datasets[pair_datasets_key] = MultitaskDatasetWrapper(
                    dataset, dataset_config.get("id", 0), 1.0, name=pair_datasets_key
                )
                size_sum += len(dataset)
                sample -= 1.0
                pair_datasets_key += "-up"

            assert sample < 1e-6, f"sample remains > 0 {pair_datasets_key}"

            logger.info(
                f"added pair {initial_pair_datasets_key} length {len(dataset)} new_length = {len(dataset)*initial_sample}"
            )
            size_by_corpus[corpus_name] += len(dataset)

        self.datasets[split] = pair_datasets
        logger.info(
            f"Datasets number = {len(self.datasets[split])} size = {size_sum} size_sum_with_subsampling = {size_sum_with_subsampling}"
        )

    @property
    def source_dictionary(self):
        return self.src_dictionary

    @property
    def target_dictionary(self):
        return self.tgt_dictionary

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):

        assert isinstance(dataset, OrderedDict)
        assert len(dataset)
        assert isinstance(dataset[next(iter(dataset))], FairseqDataset)

        # initialize the dataset with the correct starting epoch
        for _, dt in dataset.items():
            dt.set_epoch(epoch)

        indices = OrderedDict()
        batch_sampler = OrderedDict()

        with data_utils.numpy_seed(seed + epoch):
            for key, dt in dataset.items():
                logger.info(f"\t ordered_indices {key}")
                indices[key] = dt.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            for key, dt in dataset.items():
                logger.info(f"\t filter_by_size {key}")
                indices[key], ignored = dt.filter_indices_by_size(
                    indices[key], max_positions
                )

        for key, dt in dataset.items():
            logger.info(f"\t batch_by_size {key}")
            batch_sampler[key] = data_utils.batch_by_size(
                indices[key],
                dt.num_tokens,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

        epoch_iter = MultidatasetEpochBatchIterator(
            dataset=dataset,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

        return epoch_iter
