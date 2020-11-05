# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys

from fairseq.data import FileHandwritingDataset, Dictionary, AddTargetDataset, HandwritingDictionary
from . import LegacyFairseqTask, register_task


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("scribblelens")
class ScribblelensTask(LegacyFairseqTask):
    """

    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--max-sample-size",
            default=None,
            type=int,
            help="max sample size to crop to for batching. default = min sample length",
        )
        parser.add_argument(
            "--min-sample-size",
            default=None,
            type=int,
            help="min sample size to crop to for batching. default = same as --max-sample-size",
        )

        parser.add_argument(
            "--pad-to-multiples-of",
            default=None,
            type=int,
            help="enforce that lengths of inputs are multiples of this",
        )

        parser.add_argument(
            "--enable-padding",
            action="store_true",  
            help="pad shorter samples instead of cropping",  # actually needed to be set to true
        )

        parser.add_argument(
            "--labels",
            # type=bool,
            # default=None,
            type=str,
            #action="store_true",  
            help="if to return also labels from dataset" #"extension of the label file to load, if any",
        )

    def __init__(self, args, source_dictionary=None):
        super().__init__(args)
        self._target_dictionary = None
        self._source_dictionary = source_dictionary
        self.is_ctc = args.criterion == "ctc"

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if not self.args.labels:
            self.datasets[split] = FileHandwritingDataset(
                self.args.data,
                split=split,
                max_sample_size=self.args.max_sample_size,
                min_sample_size=self.args.max_sample_size,
                pad_to_multiples_of=self.args.pad_to_multiples_of,
                min_length=self.args.min_sample_size,
                pad=self.args.labels is not None or self.args.enable_padding,
                
                normalize=self.args.normalize,
            )

        else:

            # TODO change this stuff!

            #assert False  ## TODO(JCh): we must load labels from scribblelens.
            # https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md#fine-tune-a-pre-trained-model-with-ctc
            # fairseq/examples/wav2vec/libri_labels.py

            dict_path = FileHandwritingDataset.vocabularyPath(self.args.data)  #os.path.join(self.args.data, f"dict.{self.args.labels}.txt")
            self._target_dictionary = HandwritingDictionary(dict_path)  #Dictionary.load(dict_path)  

            # this dictionary ^ seems to be a file with perhaps just words? or only one occurence? or sth? 
            # seems what it does behind the hood is split the transcribed line into words and encode each word with some id, seems it assigns new ids from 0/1 for every new word it sees
            # perhaps for letters can just be letter - 'a' or sth
            # what if stuff will learn classification in a different order? need to add some additional layer or what? well, yeah, there needs to be some to predict letters from representations

            # label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")  # generated an example how this looks like
            # labels = []
            # with open(label_path, "r") as f:
            #     for line in f:
            #         labels.append(line)

            # process_label = LabelEncoder(self.target_dictionary)  // mayyybe TODO sth with that

            self.datasets[split] = FileHandwritingDataset(
                self.args.data,
                split=split,
                max_sample_size=self.args.max_sample_size,
                min_sample_size=self.args.max_sample_size,
                pad_to_multiples_of=self.args.pad_to_multiples_of,
                min_length=self.args.min_sample_size,
                pad=self.args.labels is not None or self.args.enable_padding,
                
                normalize=self.args.normalize,
                labels=True,
            )
            
            # AddTargetDataset(
            #     self.datasets[split],
            #     labels,
            #     pad=self.target_dictionary.pad(),
            #     eos=self.target_dictionary.eos(),
            #     batch_targets=True,
            #     process_label=process_label,
            #     add_to_input=not self.is_ctc,
            # )

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
            self,
            indices,
            dataset,
            max_positions=None,
            ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices
