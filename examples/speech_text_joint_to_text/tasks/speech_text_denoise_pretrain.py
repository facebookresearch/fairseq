# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import re
from argparse import Namespace
from pathlib import Path

from fairseq.data import ConcatDataset, Dictionary, encoders
from fairseq.data.audio.multi_modality_dataset import (
    FileAudioDatasetWrapper,
    ModalityDatasetItem,
    MultiModalityDataset,
)
from fairseq.data.audio.speech_to_text_joint_dataset import (
    S2TJointDataConfig,
    SpeechToTextJointDatasetCreator,
)
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.tasks import register_task

from .pair_denoising import PairedDenoisingTask

logger = logging.getLogger(__name__)


@register_task("speech_text_joint_denoising")
class SpeechTextJointDenoisingPreTask(PairedDenoisingTask):
    """
    Joint denoising training task for speech and text.
    """

    SIL_TOKEN = "sil"

    @classmethod
    def add_args(cls, parser):
        PairedDenoisingTask.add_args(parser)
        # set max tokens and position
        parser.add_argument(
            "--max-text-tokens",
            type=int,
            metavar="N",
            default=1024,
            help="maximum samples for encoder text input ",
        )
        parser.add_argument(
            "--max-speech-tokens",
            type=int,
            metavar="N",
            default=50000,
            help="maximum samples for encoder speech input ",
        )
        parser.add_argument(
            "--max-speech-positions",
            type=int,
            metavar="N",
            default=400,
            help="maximum tokens for per encoder text input ",
        )

        parser.add_argument(
            "--max-sample-size",
            type=int,
            metavar="N",
            default=32000,
            help="max sample size to crop to for batching (unsupervised speech) ",
        )
        parser.add_argument(
            "--min-sample-size",
            type=int,
            metavar="N",
            default=4000,
            help="min sample size to crop to for batching (unsupervised speech) ",
        )

        # set mini-batch ratio for different modalities/subtasks
        # s2p
        parser.add_argument(
            "--supervised-speech-sample-ratio",
            default="1",
            type=str,
            metavar="N",
            help="Multiple Ratio for speech dataset with transcripts ",
        )
        # s2t
        parser.add_argument(
            "--supervised-speech-s2s-sample-ratio",
            default="1",
            type=str,
            metavar="N",
            help="Multiple Ratio for speech dataset with transcripts ",
        )
        # ssl
        parser.add_argument(
            "--unsupervised-speech-sample-ratio",
            default="1",
            type=str,
            metavar="N",
            help="Multiple Ratio for speech dataset without transcripts ",
        )
        # t2t with monolingual data (masking)
        parser.add_argument(
            "--text-sample-ratio",
            default="1",
            type=str,
            metavar="N",
            help="Multiple Ratio for text set ",
        )
        # t2t with parallel data (no masking)
        parser.add_argument(
            "--bitext-sample-ratio",
            default="1",
            type=str,
            metavar="N",
            help="Multiple Ratio for text set (bitext) ",
        )
        # train_subset = "train", 'valid' or so
        # parallel data is loaded according to string lang_pairs and lang_pairs_no_mask from args.data
        # (un)supervised speech is loaded from args.(un)sup_speech_{train,valid}_subset
        parser.add_argument(
            "--sup-speech-data", default="", help="path to supervised speech data"
        )
        parser.add_argument(
            "--sup-speech-train-subset",
            default="",
            help="supervised speech training subsets",
        )
        parser.add_argument(
            "--sup-speech-valid-subset",
            default="",
            help="supervised speech validation subsets",
        )
        parser.add_argument(
            "--config-yaml",
            default="config.yaml",
            help="supervised speech configuration yaml file",
        )
        parser.add_argument(
            "--sup-speech-s2s-data", default="", help="path to supervised speech data"
        )
        parser.add_argument(
            "--sup-speech-s2s-train-subset",
            default="",
            help="supervised speech training subsets",
        )
        parser.add_argument(
            "--sup-speech-s2s-valid-subset",
            default="",
            help="supervised speech validation subsets",
        )
        parser.add_argument(
            "--config-s2s-yaml",
            default="config.yaml",
            help="supervised speech configuration yaml file",
        )
        parser.add_argument(
            "--unsup-speech-train-data",
            default="",
            help="path to unsupervised speech training data (tsv)",
        )
        parser.add_argument(
            "--unsup-speech-valid-data",
            default="",
            help="path to unsupervised speech valid data (tsv)",
        )
        parser.add_argument(
            "--sample-rate",
            type=int,
            metavar="N",
            default=16000,
            help="input audio sampling rate",
        )
        parser.add_argument(
            "--no-emb-update-unsup",
            default=False,
            action="store_true",
            help="no update for output embedding during unsupervised_speech mode",
        )
        parser.add_argument("--same-data-update", default=False, action="store_true")

        # used for sup_speech_ali
        parser.add_argument(
            "--use-sup-speech-ctc",
            default=False,
            action="store_true",
            help="use speech_sup_ctc instead of speech_sup_ali",
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        paths = args.data.split(":")
        assert len(paths) > 0
        src_dict = Dictionary.load(
            os.path.join(paths[0], "src_dict.txt")
        )  # assume all languages share a source dictionary
        tgt_dict = Dictionary.load(
            os.path.join(paths[0], "tgt_dict.txt")
        )  # assume all languages share a target dictionary

        lang_pairs = args.lang_pairs + "," + args.lang_pairs_bitext
        lang_pairs = re.sub(",$", "", re.sub("^,", "", lang_pairs))
        if lang_pairs != "":
            src_langs = [lp.split("-")[0] for lp in lang_pairs.split(",")]
            tgt_langs = [lp.split("-")[1] for lp in lang_pairs.split(",")]
        else:
            src_langs = []
            tgt_langs = []

        if args.add_src_lang_token:
            for lang in src_langs:
                assert (
                    src_dict.index(PairedDenoisingTask.LANG_TAG_TEMPLATE.format(lang))
                    != src_dict.unk()
                )
        if args.add_tgt_lang_token:
            for lang in tgt_langs:
                assert (
                    tgt_dict.index(PairedDenoisingTask.LANG_TAG_TEMPLATE.format(lang))
                    != tgt_dict.unk()
                )

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, src_dict, tgt_dict)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.data_cfg = S2TJointDataConfig(
            Path(args.sup_speech_data) / args.config_yaml
        )
        logger.info(
            f"load supervised speech data configure from {Path(args.sup_speech_data) / args.config_yaml}"
        )
        self.data_s2s_cfg = (
            S2TJointDataConfig(Path(args.sup_speech_s2s_data) / args.config_s2s_yaml)
            if args.sup_speech_s2s_train_subset != ""
            else None
        )
        if self.data_s2s_cfg is not None:
            logger.info(
                f"load supervised sequece to sequence speech data configure from {Path(args.sup_speech_s2s_data) / args.config_yaml}"
            )

        def parse_data_ratio(sample_ratio):
            ratios = sample_ratio.split(",")
            if len(ratios) == 1:
                return [float(ratios[0])]
            epoch_ratios = []
            for item in ratios:
                ep, r = item.split(":")
                ep = int(ep)
                r = float(r)
                assert ep > 0  # epoch is 1 based
                assert ep >= len(epoch_ratios)

                if len(epoch_ratios) == 0:
                    epoch_ratios.append(
                        r
                    )  # epoch_ratios[0] is not used, but we still set it to the first value to make thing simple.
                while len(epoch_ratios) < ep:
                    epoch_ratios.append(epoch_ratios[-1])
                epoch_ratios.append(r)
            return epoch_ratios

        self.sup_ratio = parse_data_ratio(args.supervised_speech_sample_ratio)
        self.sup_s2s_ratio = parse_data_ratio(args.supervised_speech_s2s_sample_ratio)
        self.text_ratio = parse_data_ratio(args.text_sample_ratio)
        self.bitext_ratio = parse_data_ratio(args.bitext_sample_ratio)
        self.unsup_ratio = parse_data_ratio(args.unsupervised_speech_sample_ratio)
        self.sample_mode = None

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super().build_model(args)

    def build_tokenizer(self, data_cfg, msg=""):
        logger.info(f"pre-tokenizer {msg}: {data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**data_cfg.pre_tokenizer))

    def build_bpe(self, data_cfg, msg=""):
        logger.info(f"tokenizer {msg}: {data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**data_cfg.bpe_tokenizer))

    @classmethod
    def resolve_data_type(cls, split, use_sup_speech_ctc):
        if len(split.split("_")) == 1:
            # default case, train or valid
            is_train = split
            dtype = "text"
        else:
            is_train, dtype = split.split("_", 1)
        is_train = True if is_train == "train" else False
        if dtype == "sup_speech":
            dtype = "sup_speech_ctc" if use_sup_speech_ctc else "sup_speech_ali"
        assert dtype in (
            "text",
            "bitext",
            "sup_speech_ali",
            "sup_speech_s2s",
            "unsup_speech",
            "sup_speech_ctc",
        ), f"failed resolving {split} (it resulted into: {dtype} ; is_train={is_train})"
        return is_train, dtype

    def create_modalitydatasetitem(self, dtype, dataset):
        dsitem = None
        if dtype in ("text", "bitext"):
            dsitem = ModalityDatasetItem(
                dtype,
                dataset,
                (self.args.max_source_positions, self.args.max_target_positions),
                self.args.max_text_tokens,
                self.args.batch_size,
            )
        elif dtype in ("sup_speech_ctc", "sup_speech_ali", "sup_speech_s2s"):
            dsitem = ModalityDatasetItem(
                dtype,
                dataset,
                (self.args.max_speech_positions, self.args.max_target_positions),
                self.args.max_speech_tokens,
                self.args.batch_size,
            )
        elif dtype == "unsup_speech":
            dsitem = ModalityDatasetItem(
                dtype, dataset, 1e8, self.args.max_speech_tokens, self.args.batch_size
            )
        else:
            raise ValueError(f"{dtype} is not supported")
        return dsitem

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        def _get_sup_src_tgt_dict(src_dict, tgt_dict, use_s2s_sup_decoder):
            if use_s2s_sup_decoder:
                return None, tgt_dict
            # use src_dict as tgt_dict here, since we use source dictionary as target for forcealignment
            return None, src_dict

        is_train, dtype = self.resolve_data_type(split, self.args.use_sup_speech_ctc)

        # Note we use --add-tgt-lang-token instead of data_cfg.prepend_tgt_lang_tag_no_change to set target language tag in the text dataset
        # Verify add_tgt_lang_token and prepend_tgt_lang_tag_no_change are same

        # Note we use --multilang-sampling-alpha instead of data_cfg.sampling_text_alpha to set text data sampling
        if is_train:
            msets = []
            # train split, load everything into one
            if self.lang_pairs != "":
                text_dataset = self.load_dataset_only(
                    "train", self.lang_pairs, epoch=epoch, combine=combine
                )
                dsitem = self.create_modalitydatasetitem("text", text_dataset)
                msets.append(dsitem)
            if self.lang_pairs_bitext != "":  # load bitext
                bitext_dataset = self.load_dataset_only(
                    "train_bitext",
                    self.lang_pairs_bitext,
                    do_mask=False,
                    epoch=epoch,
                    combine=combine,
                )
                dsitem = self.create_modalitydatasetitem("bitext", bitext_dataset)
                msets.append(dsitem)
            if self.args.sup_speech_train_subset != "":
                pre_tokenizer = self.build_tokenizer(self.data_cfg)
                bpe_tokenizer = self.build_bpe(self.data_cfg)

                append_eos = True
                sup_speech_type = "sup_speech_ali"
                if self.args.use_sup_speech_ctc:
                    # CTC mode
                    sup_speech_type = "sup_speech_ctc"
                    append_eos = False  # CTC doesn't need eos in the target

                src_dict, tgt_dict = _get_sup_src_tgt_dict(
                    self.src_dict, self.tgt_dict, False
                )
                sup_speech_dataset = SpeechToTextJointDatasetCreator.from_tsv(
                    self.args.sup_speech_data,
                    self.data_cfg,
                    self.args.sup_speech_train_subset,
                    tgt_dict=tgt_dict,
                    src_dict=src_dict,
                    pre_tokenizer=pre_tokenizer,
                    bpe_tokenizer=bpe_tokenizer,
                    src_pre_tokenizer=None,
                    src_bpe_tokenizer=None,
                    is_train_split=is_train,
                    epoch=epoch,
                    seed=self.args.seed,
                    append_eos=append_eos,
                )
                dsitem = self.create_modalitydatasetitem(
                    sup_speech_type, sup_speech_dataset
                )
                msets.append(dsitem)

            if self.args.sup_speech_s2s_train_subset != "":
                pre_tokenizer = self.build_tokenizer(self.data_s2s_cfg, msg="(s2s)")
                bpe_tokenizer = self.build_bpe(self.data_s2s_cfg, msg="(s2s)")

                # make sure self.data_cfg.prepend_tgt_lang_tag_no_change == self.args.add_tgt_lang_token
                src_dict, tgt_dict = _get_sup_src_tgt_dict(
                    self.src_dict, self.tgt_dict, True
                )
                sup_speech_s2s_dataset = SpeechToTextJointDatasetCreator.from_tsv(
                    self.args.sup_speech_s2s_data,
                    self.data_s2s_cfg,
                    self.args.sup_speech_s2s_train_subset,
                    tgt_dict=tgt_dict,
                    src_dict=src_dict,
                    pre_tokenizer=pre_tokenizer,
                    bpe_tokenizer=bpe_tokenizer,
                    src_pre_tokenizer=None,
                    src_bpe_tokenizer=None,
                    is_train_split=is_train,
                    epoch=epoch,
                    seed=self.args.seed,
                )
                dsitem = self.create_modalitydatasetitem(
                    "sup_speech_s2s", sup_speech_s2s_dataset
                )
                msets.append(dsitem)
            if self.args.unsup_speech_train_data != "":
                unsup_speech_dataset = FileAudioDatasetWrapper(
                    self.args.unsup_speech_train_data,
                    self.args.sample_rate,
                    max_sample_size=self.args.max_sample_size,
                    min_sample_size=self.args.min_sample_size,
                    normalize=False,
                )
                dsitem = self.create_modalitydatasetitem(
                    "unsup_speech", unsup_speech_dataset
                )
                msets.append(dsitem)

            pre_train_dataset = MultiModalityDataset(msets)
            self.datasets[split] = pre_train_dataset
        else:  # validation split, load them for each type of data
            if dtype == "text":
                text_dataset = self.load_dataset_only(
                    split, self.lang_pairs, epoch=epoch, combine=combine
                )
                dsitem = self.create_modalitydatasetitem("text", text_dataset)
                self.datasets[split] = MultiModalityDataset([dsitem])
            elif dtype == "bitext":
                bitext_dataset = self.load_dataset_only(
                    split,
                    self.lang_pairs_bitext,
                    do_mask=False,
                    epoch=epoch,
                    combine=combine,
                )
                dsitem = self.create_modalitydatasetitem("bitext", bitext_dataset)
                self.datasets[split] = MultiModalityDataset([dsitem])

            elif dtype in ("sup_speech_ctc", "sup_speech_ali"):
                assert self.args.sup_speech_valid_subset != ""
                pre_tokenizer = self.build_tokenizer(self.data_cfg)
                bpe_tokenizer = self.build_bpe(self.data_cfg)
                append_eos = True
                if dtype == "sup_speech_ctc":
                    # CTC mode
                    append_eos = False  # CTC doesn't need eos
                    assert self.args.use_sup_speech_ctc

                datasets = []
                for split_name in self.args.sup_speech_valid_subset.split(","):
                    src_dict, tgt_dict = _get_sup_src_tgt_dict(
                        self.src_dict, self.tgt_dict, False
                    )
                    datasets.append(
                        SpeechToTextJointDatasetCreator.from_tsv(
                            self.args.sup_speech_data,
                            self.data_cfg,
                            split_name,
                            tgt_dict=tgt_dict,
                            src_dict=src_dict,
                            pre_tokenizer=pre_tokenizer,
                            bpe_tokenizer=bpe_tokenizer,
                            src_pre_tokenizer=None,
                            src_bpe_tokenizer=None,
                            is_train_split=is_train,
                            epoch=epoch,
                            seed=self.args.seed,
                            append_eos=append_eos,
                        )
                    )

                dset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
                dsitem = self.create_modalitydatasetitem(dtype, dset)
                self.datasets[split] = MultiModalityDataset([dsitem])

            elif dtype == "sup_speech_s2s":
                assert self.args.sup_speech_s2s_valid_subset != ""
                pre_tokenizer = self.build_tokenizer(self.data_s2s_cfg)
                bpe_tokenizer = self.build_bpe(self.data_s2s_cfg)
                datasets = []
                for split_name in self.args.sup_speech_s2s_valid_subset.split(","):
                    src_dict, tgt_dict = _get_sup_src_tgt_dict(
                        self.src_dict, self.tgt_dict, True
                    )
                    datasets.append(
                        SpeechToTextJointDatasetCreator.from_tsv(
                            self.args.sup_speech_s2s_data,
                            self.data_s2s_cfg,
                            split_name,
                            tgt_dict=tgt_dict,
                            src_dict=src_dict,
                            pre_tokenizer=pre_tokenizer,
                            bpe_tokenizer=bpe_tokenizer,
                            src_pre_tokenizer=None,
                            src_bpe_tokenizer=None,
                            is_train_split=is_train,
                            epoch=epoch,
                            seed=self.args.seed,
                        )
                    )

                dset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
                dsitem = self.create_modalitydatasetitem("sup_speech_s2s", dset)
                self.datasets[split] = MultiModalityDataset([dsitem])
            elif dtype == "unsup_speech":
                assert self.args.unsup_speech_valid_data != ""
                unsup_speech_dataset = FileAudioDatasetWrapper(
                    self.args.unsup_speech_valid_data,
                    self.args.sample_rate,
                    max_sample_size=self.args.max_sample_size,
                    min_sample_size=self.args.min_sample_size,
                    normalize=False,
                )
                dsitem = self.create_modalitydatasetitem(
                    "unsup_speech", unsup_speech_dataset
                )
                self.datasets[split] = MultiModalityDataset([dsitem])
            else:
                raise ValueError(f"Unsupported type {dtype}")

    def get_sample_ratio(self, epoch):
        sup_ratio = (
            self.sup_ratio[epoch] if len(self.sup_ratio) > epoch else self.sup_ratio[-1]
        )
        sup_s2s_ratio = (
            self.sup_s2s_ratio[epoch]
            if len(self.sup_s2s_ratio) > epoch
            else self.sup_s2s_ratio[-1]
        )
        unsup_ratio = (
            self.unsup_ratio[epoch]
            if len(self.unsup_ratio) > epoch
            else self.unsup_ratio[-1]
        )
        text_ratio = (
            self.text_ratio[epoch]
            if len(self.text_ratio) > epoch
            else self.text_ratio[-1]
        )
        bitext_ratio = (
            self.bitext_ratio[epoch]
            if len(self.bitext_ratio) > epoch
            else self.bitext_ratio[-1]
        )
        return text_ratio, bitext_ratio, sup_ratio, sup_s2s_ratio, unsup_ratio

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
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):

        assert isinstance(dataset, MultiModalityDataset)
        if len(dataset.id_to_mode) == 1:
            max_positions = dataset.max_positions[0]
            max_tokens = dataset.max_tokens[0]
            max_sentences = dataset.max_sentences[0]
            return super().get_batch_iterator(
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
                skip_remainder_batch=skip_remainder_batch,
            )

        mult_ratio = []
        (
            text_ratio,
            bitext_ratio,
            sup_ratio,
            sup_s2s_ratio,
            unsup_ratio,
        ) = self.get_sample_ratio(epoch)
        for mode in dataset.id_to_mode:
            if mode in ("sup_speech_ctc", "sup_speech_ali"):
                mult_ratio.append(sup_ratio)
            elif mode == "sup_speech_s2s":
                mult_ratio.append(sup_s2s_ratio)
            elif mode == "text":
                mult_ratio.append(text_ratio)
            elif mode == "bitext":
                mult_ratio.append(bitext_ratio)
            elif mode == "unsup_speech":
                mult_ratio.append(unsup_ratio)

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
            mult_rate=max(self.args.update_freq) if self.args.same_data_update else 1,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter
