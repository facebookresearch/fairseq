#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq_latte_prod.data.fb_dictionary import Dictionary


@register_task('fb_translation')
class FBTranslationTask(TranslationTask):
    """
    Same as TranslationTask, Override with custom Dictionary
    """
    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)
