# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from argparse import Namespace
from pathlib import Path

import torch
from fairseq.data import (
    encoders,
    Dictionary,
    ResamplingDataset,
    TransformEosLangPairDataset,
    ConcatDataset,
)
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    LangPairMaskDataset,
    ModalityDatasetItem,
)
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset, SpeechToTextDatasetCreator
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig,
    SpeechToTextJointDatasetCreator,
)
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.tasks.translation import load_langpair_dataset

logger = logging.getLogger(__name__)
LANG_TAG_TEMPLATE = "<lang:{}>"


@register_task("speech_text_joint_to_text")
class SpeechTextJointToTextTask(SpeechToTextTask):
    """
    Task for joint training speech and text to text.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        super(SpeechTextJointToTextTask, cls).add_args(parser)
        ###
        parser.add_argument(
            "--parallel-text-data",
            default="",
            help="path to parallel text data directory",
        )
        parser.add_argument(
            "--max-tokens-text",
            type=int,
            metavar="N",
            help="maximum tokens for encoder text input ",
        )
        parser.add_argument(
            "--max-positions-text",
            type=int,
            metavar="N",
            default=400,
            help="maximum tokens for per encoder text input ",
        )
        parser.add_argument(
            "--langpairs",
            default=None,
            metavar="S",
            help='language pairs for text training, separated with ","',
        )
        parser.add_argument(
            "--speech-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for speech dataset with transcripts ",
        )
        parser.add_argument(
            "--text-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for text set ",
        )
        parser.add_argument(
            "--update-mix-data",
            action="store_true",
            help="use mixed data in one update when update-freq  > 1",
        )
        parser.add_argument(
            "--load-speech-only",
            action="store_true",
            help="load speech data only",
        )
        parser.add_argument(
            "--mask-text-ratio",
            type=float,
            metavar="V",
            default=0.0,
            help="mask V source tokens for text only mode",
        )
        parser.add_argument(
            "--mask-text-type",
            default="random",
            choices=["random", "tail"],
            help="mask text typed",
        )
        parser.add_argument(
            "--noise-token",
            default="",
            help="noise token for masking src text tokens if mask-text-ratio > 0",
        )
        parser.add_argument(
            "--infer-target-lang",
            default="",
            metavar="S",
            help="target language for inference",
        )

    def __init__(self, args, src_dict, tgt_dict, infer_tgt_lang_id=None):
        super().__init__(args, tgt_dict)
        self.src_dict = src_dict
        self.data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        assert self.tgt_dict.pad() == self.src_dict.pad()
        assert self.tgt_dict.eos() == self.src_dict.eos()
        self.speech_only = args.load_speech_only
        self._infer_tgt_lang_id = infer_tgt_lang_id

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict_path = Path(args.data) / data_cfg.vocab_filename
        src_dict_path = Path(args.data) / data_cfg.src_vocab_filename
        if (not os.path.isfile(src_dict_path)) or (not os.path.isfile(tgt_dict_path)):
            raise FileNotFoundError("Dict not found: {}".format(args.data))
        src_dict = Dictionary.load(src_dict_path.as_posix())
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())

        print("| src dictionary: {} types".format(len(src_dict)))
        print("| tgt dictionary: {} types".format(len(tgt_dict)))

        if args.parallel_text_data != "":
            if not os.path.isabs(args.parallel_text_data):
                args.parallel_text_data = os.path.join(
                    args.data, args.parallel_text_data
                )

            if args.langpairs is None:
                raise Exception(
                    "Could not infer language pair, please provide it explicitly"
                )
        infer_tgt_lang_id = None
        if args.infer_target_lang != "" and data_cfg.prepend_tgt_lang_tag_no_change:
            tgt_lang_tag = SpeechToTextDataset.LANG_TAG_TEMPLATE.format(
                args.infer_target_lang
            )
            infer_tgt_lang_id = tgt_dict.index(tgt_lang_tag)
            assert infer_tgt_lang_id != tgt_dict.unk()
        return cls(args, src_dict, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)

    def load_langpair_dataset(self, prepend_tgt_lang_tag=False, sampling_alpha=1.0, epoch=0):
        lang_pairs = []
        text_dataset = None
        split = "train"
        for lp in self.args.langpairs.split(","):
            src, tgt = lp.split("-")
            text_dataset = load_langpair_dataset(
                self.args.parallel_text_data,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=True,
                dataset_impl=None,
                upsample_primary=1,
                left_pad_source=False,
                left_pad_target=False,
                max_source_positions=self.args.max_positions_text,
                max_target_positions=self.args.max_target_positions,
                load_alignments=False,
                truncate_source=False,
            )
            if prepend_tgt_lang_tag:
                # TODO
                text_dataset = TransformEosLangPairDataset(
                    text_dataset,
                    src_eos=self.src_dict.eos(),
                    tgt_bos=self.tgt_dict.eos(),  # 'prev_output_tokens' starts with eos
                    new_tgt_bos=self.tgt_dict.index(LANG_TAG_TEMPLATE.format(tgt)),
                )
            lang_pairs.append(text_dataset)
        if len(lang_pairs) > 1:
            if sampling_alpha != 1.0:
                size_ratios = SpeechToTextDatasetCreator.get_size_ratios(
                    self.args.langpairs.split(","),
                    [len(s) for s in lang_pairs],
                    alpha=sampling_alpha,
                )
                lang_pairs = [
                    ResamplingDataset(
                        d, size_ratio=r, epoch=epoch, replace=(r >= 1.0)
                    )
                    for d, r in zip(lang_pairs, size_ratios)
                ]
            return ConcatDataset(lang_pairs)
        return text_dataset

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=self._infer_tgt_lang_id,
            )

    def build_src_tokenizer(self, args):
        logger.info(f"src-pre-tokenizer: {self.data_cfg.src_pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.src_pre_tokenizer))

    def build_src_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        ast_dataset = SpeechToTextJointDatasetCreator.from_tsv(
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
        noise_token_id = -1
        text_dataset = None
        if self.args.parallel_text_data != "" and is_train_split:
            text_dataset = self.load_langpair_dataset(
                self.data_cfg.prepend_tgt_lang_tag_no_change,
                1.0,
                epoch=epoch,
            )
            if self.args.mask_text_ratio > 0:
                # add mask
                noise_token_id = (
                    self.src_dict.unk()
                    if self.args.noise_token == ""
                    else self.src_dict.index(self.args.noise_token)
                )
                text_dataset = LangPairMaskDataset(
                    text_dataset,
                    src_bos=self.src_dict.bos(),
                    src_eos=self.src_dict.eos(),
                    noise_id=noise_token_id,
                    mask_ratio=self.args.mask_text_ratio,
                    mask_type=self.args.mask_text_type,
                )

        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "sup_speech",
                    ast_dataset,
                    (self.args.max_source_positions, self.args.max_target_positions),
                    self.args.max_tokens,
                    self.args.batch_size,
                ),
                ModalityDatasetItem(
                    "text",
                    text_dataset,
                    (self.args.max_positions_text, self.args.max_target_positions),
                    self.args.max_tokens_text
                    if self.args.max_tokens_text is not None
                    else self.args.max_tokens,
                    self.args.batch_size,
                ),
            ]
            ast_dataset = MultiModalityDataset(mdsets)
        self.datasets[split] = ast_dataset

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.tgt_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None if self.speech_only else self.src_dict

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):

        if not isinstance(dataset, MultiModalityDataset):
            return super(SpeechTextJointToTextTask, self).get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache,
            )

        mult_ratio = [self.args.speech_sample_ratio, self.args.text_sample_ratio]
        assert len(dataset.datasets) == 2

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            mult_ratio, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=1 if self.args.update_mix_data else max(self.args.update_freq),
            buffer_size=data_buffer_size,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter
