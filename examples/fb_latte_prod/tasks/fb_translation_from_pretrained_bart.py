#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.translation_from_pretrained_bart import (
    TranslationFromPretrainedBARTTask,
)
from fairseq_latte_prod.tasks.fb_translation import FBTranslationTask


@register_task("fb_translation_from_pretrained_bart")
class FBTranslationFromPretrainedBARTTask(TranslationFromPretrainedBARTTask):
    """
    Same as TranslationFromPretrainedBARTTask.
    Override dictionary with same as FBTranslationTask.
    """

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return FBTranslationTask.load_dictionary(filename)
