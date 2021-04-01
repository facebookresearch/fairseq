# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.tasks.translation import (
    TranslationTask, TranslationConfig
)

try:
    import examples.simultaneous_translation # noqa
    import_successful = True
except BaseException:
    import_successful = False


logger = logging.getLogger(__name__)


def check_import(flag):
    if not flag:
        raise ImportError(
            "'examples.simultaneous_translation' is not correctly imported. "
            "Please considering `pip install -e $FAIRSEQ_DIR`."
        )


@register_task("simul_speech_to_text")
class SimulSpeechToTextTask(SpeechToTextTask):
    def __init__(self, args, tgt_dict):
        check_import(import_successful)
        super().__init__(args, tgt_dict)


@register_task("simul_text_to_text",  dataclass=TranslationConfig)
class SimulTextToTextTask(TranslationTask):
    def __init__(self, cfg, src_dict, tgt_dict):
        check_import(import_successful)
        super().__init__(cfg, src_dict, tgt_dict)
