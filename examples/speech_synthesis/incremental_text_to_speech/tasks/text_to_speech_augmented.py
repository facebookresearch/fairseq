# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq.tasks import register_task
from fairseq.tasks.text_to_speech import TextToSpeechTask
from examples.speech_synthesis.incremental_text_to_speech.model.text_to_speech import prefix_augmenter
import random


logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)


try:
    from tensorboardX import SummaryWriter
except ImportError:
    logger.info("Please install tensorboardX: pip install tensorboardX")
    SummaryWriter = None


@register_task('text_to_speech_augmented')
class TextToSpeechTaskAugmented(TextToSpeechTask):
    @staticmethod
    def add_args(parser):
        TextToSpeechTask.add_args(parser)
        parser.add_argument('--prefix-probability', type=float, default=0.5,
                            help='Probability of replacing a minibatch by prefixes')
        parser.add_argument('--prefix-augment-ratio', type=int, default=3,
                            help='Number different lengths. When prefix_augment_ratio is 3,'
                                 'training samples will be augmented to 1/3 or 2/3 the length of their original length.')

    def __init__(self, args, src_dict):
        super().__init__(args, src_dict)
        self.augmenter = prefix_augmenter.PartialSequenceAugmenter(src_dict)
        self.prefix_probability = args.prefix_probability
        self.prefix_augment_ratio = args.prefix_augment_ratio

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        if random.random() < self.prefix_probability:
            # Train with prefixes
            augmented_sample = self.augmenter.augment(sample, self.prefix_augment_ratio)

            partial_seq_loss, partial_seq_sample_size, partial_seq_logging_output = \
                super().train_step(augmented_sample, model, criterion, optimizer, update_num, ignore_grad)

            return partial_seq_loss, partial_seq_sample_size, partial_seq_logging_output
        else:
            # Train with original samples
            return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
