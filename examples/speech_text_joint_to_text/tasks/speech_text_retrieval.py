# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from fairseq.data.audio.speech_to_text_joint_dataset import SpeechToTextJointDatasetCreator
from fairseq.tasks import register_task

from examples.speech_text_joint_to_text.data.retrieval_wrapper_dataset import SpeechTextRetrievalDataset
from .speech_text_joint import SpeechTextJointToTextTask


logger = logging.getLogger(__name__)


@register_task("speech_text_retrieval")
class SpeechTextRetrievalTask(SpeechTextJointToTextTask):
    def __init__(self, args, src_dict, tgt_dict, infer_tgt_lang_id=None):
        super().__init__(args, src_dict, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)
        self.min_source_len = args.min_source_len

    @classmethod
    def add_args(cls, parser):
        super(SpeechTextRetrievalTask, cls).add_args(parser)
        parser.add_argument(
            "--min-source-len",
            type=int,
            metavar="N",
            default=15000,
            help="minimun length for encoder speech input ",
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        self.datasets[split] = SpeechTextRetrievalDataset(SpeechToTextJointDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            src_dict=None if self.speech_only else self.src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        ), self.src_dict)

    def filter_indices_by_size(self, indices, dataset, max_positions=None, ignore_invalid_inputs=False):
        indices = super().filter_indices_by_size(indices, dataset, max_positions, ignore_invalid_inputs)
        if hasattr(dataset, 'filter_short_utterances'):
            indices = dataset.filter_short_utterances(indices, self.min_source_len, ignore_invalid_inputs)
        return indices
