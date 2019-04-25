# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from fairseq.data.masked_lm_dictionary import MaskedLMDictionary
from fairseq.tasks.translation import TranslationTask

from . import register_task


@register_task("translation_from_pretrained_xlm")
class TranslationFromPretrainedXLMTask(TranslationTask):
    """
    Same as TranslationTask except use the MaskedLMDictionary class so that
    we can load data that was binarized with the MaskedLMDictionary class.

    This task should be used for the entire training pipeline when we want to
    train an NMT model from a pretrained XLM checkpoint: binarizing NMT data,
    training NMT with the pretrained XLM checkpoint, and subsequent evaluation
    of that trained model.
    """

    @classmethod
    def load_dictionary(cls, filename):
        """Load the masked LM dictionary from the filename

        Args:
            filename (str): the filename
        """
        return MaskedLMDictionary.load(filename)
