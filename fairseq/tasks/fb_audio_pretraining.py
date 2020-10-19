# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
from itertools import count

from fairseq.data.audio.fb_everstore_audio_dataset import EverstoreAudioDataset

from . import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("fb_audio_pretraining")
class SpeechPretrainingTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=int,
            help="target sample rate. audio files will be up/down sampled to this rate",
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

    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)

    def create_dataset(self, manifest):
        logger.info("Reading manifest " + manifest)
        return EverstoreAudioDataset(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.min_sample_size,
            min_length=16000,
        )

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        if os.path.isfile(manifest):
            self.datasets[split] = self.create_dataset(manifest)
        else:
            if not hasattr(self, "dataset_parts"):
                self.dataset_parts = []
                for i in count(1):
                    manifest = os.path.join(self.args.data, "{}{}.tsv".format(split, i))
                    if not os.path.isfile(manifest):
                        break
                    self.dataset_parts.append(manifest)
                if len(self.dataset_parts) == 0:
                    raise FileNotFoundError(manifest)
            part_idx = (epoch - 1) % len(self.dataset_parts)
            self.datasets[split] = self.create_dataset(self.dataset_parts[part_idx])

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return None
