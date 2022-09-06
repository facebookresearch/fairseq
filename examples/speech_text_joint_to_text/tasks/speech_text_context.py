# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from fairseq.data.concat_dataset import ConcatDataset

from fairseq.tasks import register_task

from examples.speech_text_joint_to_text.data.context_wrapper_dataset import SpeechTextContextDataset
from examples.speech_text_joint_to_text.data.speech_to_text_joint_dataset_with_entities import SpeechToTextJointWithEntitiesDatasetCreator
from .speech_text_retrieval import SpeechTextRetrievalTask


logger = logging.getLogger(__name__)


@register_task("speech_text_context")
class SpeechTextContextTask(SpeechTextRetrievalTask):
    def __init__(self, args, src_dict, tgt_dict, infer_tgt_lang_id=None):
        super().__init__(args, src_dict, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)
        if SpeechTextContextDataset.CONTEXT_TAG not in tgt_dict:
            tgt_dict.add_symbol(SpeechTextContextDataset.CONTEXT_TAG)

    @classmethod
    def add_args(cls, parser):
        super(SpeechTextContextTask, cls).add_args(parser)
        parser.add_argument(
            "--max-positives",
            type=int,
            metavar="N",
            default=3,
            help="number of positives words per sample",
        )

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        ds_with_entities = SpeechToTextJointWithEntitiesDatasetCreator.from_tsv(
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
        )
        if isinstance(ds_with_entities, ConcatDataset):
            for d in ds_with_entities.datasets:
                d.source_entities = False
        else:
            ds_with_entities.source_entities = False
        self.datasets[split] = SpeechTextContextDataset(
            ds_with_entities,
            self.tgt_dict,
            num_negatives=self.args.num_negatives,
            max_words=self.args.max_words,
            max_positives=self.args.max_positives)
