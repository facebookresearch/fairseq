# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from fairseq.data import RawAudioDataset
from . import FairseqTask, register_task


@register_task('audio_pretraining')
class AudioPretrainingTask(FairseqTask):
    """

    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-rate', default=16000, type=int,
                            help='target sample rate. audio files will be up/down sampled to this rate')
        parser.add_argument('--max-sample-size', default=None, type=int,
                            help='max sample size to crop to for batching. default = min sample length')
        parser.add_argument('--min-sample-size', default=None, type=int,
                            help='min sample size to crop to for batching. default = same as --max-sample-size')

    def __init__(self, args):
        super().__init__(args)

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

        manifest = os.path.join(self.args.data, '{}.tsv'.format(split))
        self.datasets[split] = RawAudioDataset(manifest,
                                               sample_rate=self.args.sample_rate,
                                               max_sample_size=self.args.max_sample_size,
                                               min_sample_size=self.args.min_sample_size)

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return None
