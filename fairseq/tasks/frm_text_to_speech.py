# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq.data.audio.frm_text_to_speech_dataset import FrmTextToSpeechDatasetCreator
from fairseq.tasks import register_task
from fairseq.tasks.text_to_speech import TextToSpeechTask


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@register_task("frm_text_to_speech")
class FrmTextToSpeechTask(TextToSpeechTask):
    @staticmethod
    def add_args(parser):
        TextToSpeechTask.add_args(parser)
        parser.add_argument("--do_chunk", action="store_true", help="train on chunks")
        parser.add_argument("--chunk_bound", default=-1, type=int)
        parser.add_argument("--chunk_init", default=50, type=int)
        parser.add_argument("--chunk_incr", default=5, type=int)
        parser.add_argument("--add_eos", action="store_true")
        parser.add_argument("--dedup", action="store_true")
        parser.add_argument("--ref_fpu", default=-1, type=float)

    def load_dataset(self, split, **unused_kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = FrmTextToSpeechDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            n_frames_per_step=self.args.n_frames_per_step,
            speaker_to_id=self.speaker_to_id,
            do_chunk=self.args.do_chunk,
            chunk_bound=self.args.chunk_bound,
            chunk_init=self.args.chunk_init,
            chunk_incr=self.args.chunk_incr,
            add_eos=self.args.add_eos,
            dedup=self.args.dedup,
            ref_fpu=self.args.ref_fpu,
        )
