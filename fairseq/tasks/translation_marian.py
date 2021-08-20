# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from fairseq.data import (
    TokenizerDictionary,
)
from fairseq.tasks import register_task
from .translation import TranslationTask


logger = logging.getLogger(__name__)


@register_task('translation_marian')
class HuggingFaceTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--traced', action='store_true',
                            help='whether to use traced model or not')

    def __init__(self, args, data_dictionary):
        super().__init__(args, data_dictionary, data_dictionary)
        self.dictionary = data_dictionary
        self.tokenizer = data_dictionary.tokenizer
        self.args = args

    @classmethod
    def load_dictionary(cls,  model_path):
        dictionary = TokenizerDictionary.load(model_path)
        return dictionary

    @classmethod
    def setup_task(cls, cfg, **kwargs):

        # load data dictionary
        data_dict = cls.load_dictionary(
            cfg.path.split(':')[0]
        )
        return HuggingFaceTranslationTask(cfg, data_dict)


    def build_model(self, cfg):
        from fairseq import models
        logger.info(cfg)
        if self.args.traced:
            cfg._name = 'hf_marian_traced'
        else:
            cfg._name = 'hf_marian'
        model = models.build_model(cfg, self)
        return model


    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary

    def max_positions(self):
        return (512, 512)
