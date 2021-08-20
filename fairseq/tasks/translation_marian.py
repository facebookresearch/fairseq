# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from fairseq.data import (
    data_utils,
    TokenizerDictionary,
    RawLabelDataset,
    MarianTokenizerDataset,
)
from fairseq.tasks import FairseqTask,  register_task
from .translation import TranslationTask, TranslationConfig


logger = logging.getLogger(__name__)


@register_task('translation_marian')
class HuggingFaceTranslationTask(TranslationTask):
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


    def build_model(self, cfg, name):
        from fairseq import models
        #logger.info(cfg)
        cfg._name == 'name'
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
