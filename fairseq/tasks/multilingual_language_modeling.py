# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    ConcatDataset,
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


def lang_token(lang):
    return f"<{lang}>"


@dataclass
class MultilingualLanguageModelingConfig(FairseqDataclass):
    # TODO common var add to parent
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend lang id token <dialect>"}
    )
    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False, metadata={"help": "pad to fixed length"}
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False, metadata={"help": "boolean to pad to fixed batch size"}
    )

    multilang_sampling_alpha: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "smoothing alpha for sample rations across multiple datasets"
        },
    )

    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )

    langs: str = field(
        default="",
        metadata={
            "help": "comma-separated list of languages (default: all directories in data path)"
        },
    )
    baseline_model_langs: str = field(
        default="",
        metadata={
            "help": "comma-separated list of languages in the baseline model (default: none)"
        },
    )
    # TODO: legacy parameter kept for compatibility
    baseline_model: str = field(
        default="",
        metadata={
            "help": "path to the baseline model (default: none)"
        },
    )
    
    lang_to_offline_shard_ratio: str = field(
        default="",
        metadata={
            "help": "absolute path of tsv file location to indicate lang to offline shard ratio.",
        }
    )
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    train_subset: str = II("common.train_subset")
    valid_subset: str = II("common.valid_subset")


@register_task("multilingual_language_modeling", dataclass=MultilingualLanguageModelingConfig)
class MultilingualLanguageModelingTask(LegacyFairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary

        if targets is None:
            targets = ["future"]
        self.targets = targets

    @staticmethod
    def _get_langs(args, epoch=1):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        languages = sorted(
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        )
        if args.langs:
            keep_langs = set(args.langs.split(","))
            languages = [lang for lang in languages if lang in keep_langs]
            assert len(languages) == len(keep_langs)

        return languages, data_path

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            if args.add_bos_token:
                languages, _ = cls._get_langs(args)
                logger.info('----------------')
                for lang in languages:
                    dictionary.add_symbol(lang_token(lang))
                    logger.info(f'add language token: {lang_token(lang)}')
                logger.info('----------------')

            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )
        return (dictionary, output_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)

        # upgrade old checkpoints
        if hasattr(args, "exclude_self_target"):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    f"Unsupported language modeling target: {target} not in {model.supported_targets}"
                )

        return model

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        languages, data_path = MultilingualLanguageModelingTask._get_langs(
            self.args, epoch
        )
        lang_to_offline_shard_ratio = None
        if self.args.lang_to_offline_shard_ratio != "":
            lang_to_offline_shard_ratio = {}
            assert os.path.exists(
                self.args.lang_to_offline_shard_ratio
            ), "provided offline shard ratio file doesn't exist: {0}".format(self.args.lang_to_offline_shard_ratio)
            with open(self.args.lang_to_offline_shard_ratio) as fin:
                for line in fin:
                    lang, ratio = line.strip().split('\t')
                    ratio = float(ratio)
                    lang_to_offline_shard_ratio[lang] = ratio
           
            logger.info(
                "Found offline sharded ratio: %s", lang_to_offline_shard_ratio,
            )

        if split == self.args.train_subset:
            logger.info("Training on {0} languages: {1}".format(len(languages), languages))
        else:
            logger.info("Evaluating on {0} languages: {1}".format(len(languages), languages))

        tokens_per_sample = self.args.tokens_per_sample - int(self.args.add_bos_token)

        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = (
                self.args.batch_size_valid if "valid" in split else self.args.batch_size
            )

        lang_datasets = []
        for lang_id, language in enumerate(languages):
            split_path = os.path.join(data_path, language, split)
            dataset = data_utils.load_indexed_dataset(
                split_path, self.dictionary, self.args.dataset_impl, combine=combine
            )
            # print('len(dataset) =', len(dataset))
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            dataset = maybe_shorten_dataset(
                dataset,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                tokens_per_sample,
                self.args.seed,
            )

            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode=self.args.sample_break_mode,
                include_targets=True,
            )

            add_eos_for_other_targets = (
                self.args.sample_break_mode is not None
                and self.args.sample_break_mode != "none"
            )
            src_lang_idx, tgt_lang_idx = None, None
            if self.args.add_bos_token:
                src_lang_idx = self.dictionary.index(lang_token(language))
                tgt_lang_idx = self.output_dictionary.index(lang_token(language))

            lang_datasets.append(
                MonolingualDataset(
                    dataset=dataset,
                    sizes=dataset.sizes,
                    src_vocab=self.dictionary,
                    tgt_vocab=self.output_dictionary,
                    add_eos_for_other_targets=add_eos_for_other_targets,
                    shuffle=True,
                    targets=self.targets,
                    fixed_pad_length=fixed_pad_length,
                    pad_to_bsz=pad_to_bsz,
                    add_bos_token=self.args.add_bos_token,
                    src_lang_idx=src_lang_idx,
                    tgt_lang_idx=tgt_lang_idx,
                )
            )

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            "loaded total {} blocks for all languages".format(
                dataset_lengths.sum(),
            )
        )
        if split == self.args.train_subset:
            dataset_lengths_ratio_multiplier = np.ones(len(dataset_lengths))
            if lang_to_offline_shard_ratio is not None:            
                dataset_lengths_ratio_multiplier = []
                for lang in languages:
                    assert lang in lang_to_offline_shard_ratio, "Lang: {0} missing in offline shard ratio file: {1}".format(
                        lang, self.args.lang_to_offline_shard_ratio,
                    )
                    dataset_lengths_ratio_multiplier.append(lang_to_offline_shard_ratio[lang])
                dataset_lengths_ratio_multiplier = np.array(dataset_lengths_ratio_multiplier)
                true_dataset_lengths = dataset_lengths * dataset_lengths_ratio_multiplier
            else:
                true_dataset_lengths = dataset_lengths
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(true_dataset_lengths)

            logger.info(
                "Sample probability by language: %s",
                {
                    lang: "{0:.4f}".format(sample_probs[id])
                    for id, lang in enumerate(languages)
                },
            )
            size_ratio = (sample_probs * true_dataset_lengths.sum()) / dataset_lengths
            # TODO: add an option for shrinking all size ratios to below 1 
            # if self.args.multilang_sampling_alpha != 1:
            #   size_ratio /= size_ratio.max()

            # Fix numeric errors in size ratio computation
            #   0.999999999999999999 -> 1
            #   1.000000000000000002 -> 1
            for i in range(len(size_ratio)):
                size_ratio[i] = round(size_ratio[i], 8) 

            logger.info(
                "Up/Down Sampling ratio by language: %s",
                {
                    lang: "{0:.2f}".format(size_ratio[id])
                    for id, lang in enumerate(languages)
                },
            )
            logger.info(
                "Actual dataset size by language: %s",
                {
                    lang: "{0:.2f}".format(len(lang_datasets[id]))
                    for id, lang in enumerate(languages)
                },
            )
            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] > 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            logger.info(
                "Resampled dataset size by language: %s",
                {
                    lang: "{0:.2f}".format(len(resampled_lang_datasets[id]))
                    for id, lang in enumerate(languages)
                },
            )
            dataset = ConcatDataset(resampled_lang_datasets)
        else:
            dataset = ConcatDataset(lang_datasets)
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + "_" + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            # [TODO]: This is hacky for now to print validation ppl for each
            # language individually. Maybe need task API changes to allow it
            # in more generic ways.
            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ",".join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, language="en_XX", **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )

        src_lang_idx = self.dictionary.index(lang_token(language))
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                (src_lang_idx or self.source_dictionary.bos())
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )

        max_seq_len = max(src_lengths) + 1
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                        pad_length=max_seq_len
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False, pad_length=max_seq_len,
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    @torch.no_grad()
    def inference_step(
        self, generator, models, sample, language="en_XX", prefix_tokens=None, constraints=None
    ):
        # Generation will always be conditioned on bos_token
        if getattr(self.args, "add_bos_token", False):
            src_lang_idx = self.dictionary.index(lang_token(language))
            bos_token = src_lang_idx or self.source_dictionary.bos()
        else:
            bos_token = self.source_dictionary.eos()

        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the language_modeling task is not supported"
            )

        # SequenceGenerator doesn't use src_tokens directly, we need to
        # pass the `prefix_tokens` argument instead
        if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
            prefix_tokens = sample["net_input"]["src_tokens"]
            if prefix_tokens[:, 0].eq(bos_token).all():
                prefix_tokens = prefix_tokens[:, 1:]

        return generator.generate(
            models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
        )
            
    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        )

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary